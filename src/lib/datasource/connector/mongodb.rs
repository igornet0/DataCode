//! MongoDB DataSource connector: documents → Normalizer → Table.

use crate::common::numeric::IntValue;
use crate::common::table::Table;
use crate::common::value::{ByteBuffer, ObjectKind, Value};
use crate::datasource::capabilities::Capabilities;
use crate::datasource::config::{sql_url_from_config, DataSourceConfig};
use crate::datasource::connector::ConnectorBackend;
use crate::datasource::error::DataSourceError;
use crate::datasource::normalize::{
    documents_to_table, filter_ops_to_mongo_json, parse_filter_ops, FlattenOptions,
};
use crate::datasource::request::{GetTableSpec, RequestSpec, SendTableSpec};
use crate::datasource::response::DataSourceResponse;
use crate::web::browser::runtime::block_on;
use bson::{Bson, Document};
use chrono::{FixedOffset, TimeZone, Utc};
use futures_util::TryStreamExt;
use mongodb::options::{ClientOptions, FindOptions};
use mongodb::{Client, Collection, Database};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

pub struct MongoConnector {
    config: DataSourceConfig,
    client: Option<Client>,
    url: String,
}

impl MongoConnector {
    pub fn new(config: DataSourceConfig) -> Result<Self, DataSourceError> {
        let url = sql_url_from_config(&config).or_else(|_| {
            config.url.clone().ok_or_else(|| DataSourceError::Validation {
                message: "mongodb datasource requires 'url'/'uri' or host/database".to_string(),
            })
        })?;
        if config.database.is_none() {
            // Allow database in URL path; still warn via validation if completely missing later.
        }
        Ok(Self {
            config,
            client: None,
            url,
        })
    }

    fn database_name(&self) -> Result<&str, DataSourceError> {
        self.config
            .database
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| DataSourceError::Validation {
                message: "MongoDatabaseNotFound: mongodb datasource requires 'database'".to_string(),
            })
    }

    fn collection_name<'a>(
        &'a self,
        spec_collection: Option<&'a str>,
    ) -> Result<&'a str, DataSourceError> {
        spec_collection
            .or(self.config.collection.as_deref())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| DataSourceError::Validation {
                message: "MongoCollectionNotFound: collection required in config or spec"
                    .to_string(),
            })
    }

    fn client(&mut self) -> Result<&Client, DataSourceError> {
        if self.client.is_none() {
            self.connect()?;
        }
        Ok(self.client.as_ref().unwrap())
    }

    fn db(&mut self) -> Result<Database, DataSourceError> {
        let name = self.database_name()?.to_string();
        let client = self.client()?;
        Ok(client.database(&name))
    }

    fn collection(&mut self, name: &str) -> Result<Collection<Document>, DataSourceError> {
        Ok(self.db()?.collection(name))
    }

    fn build_filter_doc(spec_filter: Option<&Value>, native: Option<&Value>) -> Result<Document, DataSourceError> {
        if let Some(n) = native {
            return value_to_document(n);
        }
        if let Some(f) = spec_filter {
            // If filter looks like raw Mongo (`$and` / `$or` / field with `$gt`), pass through.
            if is_raw_mongo_filter(f) {
                return value_to_document(f);
            }
            let ops = parse_filter_ops(f).map_err(|e| DataSourceError::Validation {
                message: format!("MongoQueryError: {}", e),
            })?;
            let json = filter_ops_to_mongo_json(&ops);
            return json_to_document(&json);
        }
        Ok(Document::new())
    }

    fn flatten_opts(spec: &GetTableSpec) -> FlattenOptions {
        FlattenOptions::from_array_mode_str(spec.array_mode.as_deref())
    }

    fn fetch_documents(
        &mut self,
        collection: &str,
        filter: Document,
        sort: Option<Document>,
        projection: Option<Document>,
        limit: Option<usize>,
        offset: Option<usize>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Value>, DataSourceError> {
        let coll = self.collection(collection)?;
        let mut opts = FindOptions::default();
        if let Some(l) = limit {
            opts.limit = Some(l as i64);
        }
        if let Some(o) = offset {
            opts.skip = Some(o as u64);
        }
        if let Some(s) = sort {
            opts.sort = Some(s);
        }
        if let Some(p) = projection {
            opts.projection = Some(p);
        }
        if let Some(bs) = batch_size {
            opts.batch_size = Some(bs as u32);
        } else {
            opts.batch_size = Some(1000);
        }
        block_on(async {
            let mut cursor = coll
                .find(filter)
                .with_options(opts)
                .await
                .map_err(map_mongo_err)?;
            let mut docs = Vec::new();
            while let Some(doc) = cursor.try_next().await.map_err(map_mongo_err)? {
                docs.push(bson_document_to_value(&doc));
            }
            Ok(docs)
        })
    }

    fn run_aggregation(
        &mut self,
        collection: &str,
        pipeline: Vec<Document>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Value>, DataSourceError> {
        let coll = self.collection(collection)?;
        block_on(async {
            let mut cursor = coll
                .aggregate(pipeline)
                .batch_size(batch_size.unwrap_or(1000) as u32)
                .await
                .map_err(map_mongo_err)?;
            let mut docs = Vec::new();
            while let Some(doc) = cursor.try_next().await.map_err(map_mongo_err)? {
                docs.push(bson_document_to_value(&doc));
            }
            Ok(docs)
        })
    }
}

impl ConnectorBackend for MongoConnector {
    fn connector_type(&self) -> &str {
        "mongodb"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::mongodb_default()
    }

    fn connect(&mut self) -> Result<(), DataSourceError> {
        if self.client.is_some() {
            return Ok(());
        }
        let mut url = self.url.clone();
        // Inject credentials into URI if provided separately.
        if let (Some(user), Some(pass)) = (&self.config.username, &self.config.password) {
            if !url.contains('@') && url.starts_with("mongodb") {
                if let Some(rest) = url.strip_prefix("mongodb://") {
                    url = format!("mongodb://{}:{}@{}", user, pass, rest);
                } else if let Some(rest) = url.strip_prefix("mongodb+srv://") {
                    url = format!("mongodb+srv://{}:{}@{}", user, pass, rest);
                }
            }
        }
        let timeout = self.config.timeout.unwrap_or(30.0);
        block_on(async {
            let mut opts = ClientOptions::parse(&url)
                .await
                .map_err(|e| DataSourceError::Connection {
                    message: format!("MongoConnectionError: {}", e),
                })?;
            opts.server_selection_timeout = Some(Duration::from_secs_f64(timeout.max(0.1)));
            opts.connect_timeout = Some(Duration::from_secs_f64(
                self.config.connect_timeout.unwrap_or(timeout).max(0.1),
            ));
            let client = Client::with_options(opts).map_err(|e| DataSourceError::Connection {
                message: format!("MongoConnectionError: {}", e),
            })?;
            // Force handshake.
            client
                .list_database_names()
                .await
                .map_err(map_mongo_err)?;
            self.client = Some(client);
            Ok(())
        })
    }

    fn disconnect(&mut self) {
        self.client = None;
    }

    fn ping(&mut self) -> Result<bool, DataSourceError> {
        let client = self.client()?;
        block_on(async {
            client
                .database("admin")
                .run_command(doc! { "ping": 1 })
                .await
                .map(|_| true)
                .map_err(map_mongo_err)
        })
    }

    fn test(&mut self) -> Result<Value, DataSourceError> {
        match self.ping() {
            Ok(ok) => Ok(mongo_diagnostic(ok, &self.url, self.config.database.as_deref(), None)),
            Err(e) => Ok(mongo_diagnostic(
                false,
                &self.url,
                self.config.database.as_deref(),
                Some(e.display()),
            )),
        }
    }

    fn request(&mut self, spec: &RequestSpec) -> Result<DataSourceResponse, DataSourceError> {
        let start = Instant::now();
        let op = spec
            .op
            .as_deref()
            .unwrap_or("find")
            .to_ascii_lowercase();
        let body = match op.as_str() {
            "count" => {
                let coll_name = self.collection_name(spec.collection.as_deref())?.to_string();
                let filter = Self::build_filter_doc(spec.filter.as_ref(), spec.native.as_ref())?;
                let coll = self.collection(&coll_name)?;
                let n = block_on(async {
                    coll.count_documents(filter)
                        .await
                        .map_err(map_mongo_err)
                })?;
                format!("{{\"count\": {}}}", n)
            }
            "list_databases" => {
                let client = self.client()?;
                let names = block_on(async {
                    client.list_database_names().await.map_err(map_mongo_err)
                })?;
                serde_json::to_string(&serde_json::json!({ "databases": names })).unwrap_or_default()
            }
            "list_collections" => {
                let db = self.db()?;
                let names = block_on(async {
                    db.list_collection_names().await.map_err(map_mongo_err)
                })?;
                serde_json::to_string(&serde_json::json!({ "collections": names }))
                    .unwrap_or_default()
            }
            "delete" | "delete_many" => {
                let coll_name = self.collection_name(spec.collection.as_deref())?.to_string();
                // Empty filter `{}` deletes all documents in the collection.
                let filter = Self::build_filter_doc(spec.filter.as_ref(), spec.native.as_ref())?;
                let coll = self.collection(&coll_name)?;
                let n = block_on(async {
                    coll.delete_many(filter)
                        .await
                        .map_err(map_mongo_err)
                        .map(|r| r.deleted_count)
                })?;
                format!("{{\"deleted\": {}}}", n)
            }
            "drop" | "drop_collection" => {
                let coll_name = self.collection_name(spec.collection.as_deref())?.to_string();
                let coll = self.collection(&coll_name)?;
                block_on(async { coll.drop().await.map_err(map_mongo_err) })?;
                format!("{{\"dropped\": \"{}\"}}", coll_name)
            }
            "find" | _ => {
                let coll_name = self.collection_name(spec.collection.as_deref())?.to_string();
                let filter = Self::build_filter_doc(spec.filter.as_ref(), spec.native.as_ref())?;
                let docs = if let Some(agg) = &spec.aggregation {
                    let pipeline = value_to_pipeline(agg)?;
                    self.run_aggregation(&coll_name, pipeline, None)?
                } else {
                    self.fetch_documents(
                        &coll_name,
                        filter,
                        None,
                        None,
                        spec.limit,
                        spec.offset,
                        None,
                    )?
                };
                format!("{{\"documents\": {}}}", docs.len())
            }
        };
        Ok(DataSourceResponse::new(
            200,
            self.url.clone(),
            HashMap::from([(
                "Content-Type".to_string(),
                "application/json".to_string(),
            )]),
            body.into_bytes(),
            start.elapsed(),
        ))
    }

    fn get_table(&mut self, spec: &GetTableSpec) -> Result<Table, DataSourceError> {
        let coll_name = self.collection_name(spec.collection.as_deref())?.to_string();
        let opts = Self::flatten_opts(spec);
        let docs = if let Some(agg) = &spec.aggregation {
            let pipeline = value_to_pipeline(agg)?;
            self.run_aggregation(&coll_name, pipeline, spec.batch_size)?
        } else {
            let filter = Self::build_filter_doc(spec.filter.as_ref(), spec.native.as_ref())?;
            let sort = spec
                .sort
                .as_ref()
                .map(value_to_document)
                .transpose()?;
            let projection = spec
                .select
                .as_ref()
                .map(value_to_document)
                .transpose()?;
            self.fetch_documents(
                &coll_name,
                filter,
                sort,
                projection,
                spec.limit,
                spec.offset,
                spec.batch_size,
            )?
        };
        Ok(documents_to_table(&docs, &opts))
    }

    fn send_table(&mut self, spec: &SendTableSpec) -> Result<(), DataSourceError> {
        if spec.mode != "append" {
            return Err(DataSourceError::Unsupported {
                message: format!(
                    "send_table mode '{}' not supported for mongodb (use 'append')",
                    spec.mode
                ),
            });
        }
        let coll_name = self
            .collection_name(spec.collection.as_deref().or(spec.table_name.as_deref()))?
            .to_string();
        let table_val = spec.table.as_ref().ok_or_else(|| DataSourceError::Validation {
            message: "send_table requires table".to_string(),
        })?;
        let Value::Table(rc) = table_val else {
            return Err(DataSourceError::Validation {
                message: "send_table requires a table value".to_string(),
            });
        };
        let table = rc.borrow();
        let headers = table.headers().clone();
        if headers.is_empty() {
            return Ok(());
        }
        let rows = table.rows_ref().map(|r| r.to_vec()).unwrap_or_default();
        let mut docs = Vec::with_capacity(rows.len());
        for row in rows {
            let mut doc = Document::new();
            for (i, h) in headers.iter().enumerate() {
                let v = row.get(i).cloned().unwrap_or(Value::Null);
                doc.insert(h.clone(), value_to_bson(&v));
            }
            docs.push(doc);
        }
        let coll = self.collection(&coll_name)?;
        if docs.is_empty() {
            return Ok(());
        }
        block_on(async {
            coll.insert_many(docs).await.map_err(map_mongo_err)?;
            Ok(())
        })
    }

    fn clone_backend(&self) -> Box<dyn ConnectorBackend> {
        Box::new(Self {
            config: self.config.clone(),
            client: None,
            url: self.url.clone(),
        })
    }
}

fn map_mongo_err(e: mongodb::error::Error) -> DataSourceError {
    let msg = e.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("auth") {
        DataSourceError::Authentication {
            message: format!("MongoAuthenticationError: {}", msg),
        }
    } else if lower.contains("timeout") {
        DataSourceError::Timeout {
            message: format!("MongoTimeoutError: {}", msg),
        }
    } else if lower.contains("not found") {
        DataSourceError::NotFound {
            message: format!("MongoNotFound: {}", msg),
        }
    } else {
        DataSourceError::Other {
            message: format!("MongoQueryError: {}", msg),
        }
    }
}

fn mongo_diagnostic(
    ok: bool,
    url: &str,
    database: Option<&str>,
    message: Option<String>,
) -> Value {
    let mut m = HashMap::new();
    m.insert("ok".to_string(), Value::Bool(ok));
    m.insert("type".to_string(), Value::String("mongodb".to_string()));
    // Never echo password — redact userinfo.
    m.insert("url".to_string(), Value::String(redact_url(url)));
    if let Some(db) = database {
        m.insert("database".to_string(), Value::String(db.to_string()));
    }
    if let Some(msg) = message {
        m.insert("message".to_string(), Value::String(msg));
    } else if ok {
        m.insert("message".to_string(), Value::String("ok".to_string()));
    }
    Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(m))))
}

fn redact_url(url: &str) -> String {
    if let Some(scheme_end) = url.find("://") {
        let scheme = &url[..scheme_end + 3];
        let rest = &url[scheme_end + 3..];
        if let Some(at) = rest.find('@') {
            return format!("{}***@{}", scheme, &rest[at + 1..]);
        }
    }
    url.to_string()
}

fn is_raw_mongo_filter(v: &Value) -> bool {
    let Value::Object(rc) = v else {
        return false;
    };
    let kind = rc.borrow();
    let keys: Vec<String> = match &*kind {
        ObjectKind::Legacy(m) => m.keys().cloned().collect(),
        ObjectKind::Inline(entries) => entries
            .iter()
            .filter_map(|(k, _)| match k {
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        ObjectKind::Bucket(_) => return false,
    };
    keys.iter().any(|k| k.starts_with('$'))
        || keys.iter().any(|k| {
            if let ObjectKind::Legacy(m) = &*kind {
                if let Some(Value::Object(inner)) = m.get(k) {
                    let ik = inner.borrow();
                    return match &*ik {
                        ObjectKind::Legacy(im) => im.keys().any(|x| x.starts_with('$')),
                        _ => false,
                    };
                }
            }
            false
        })
}

fn value_to_pipeline(v: &Value) -> Result<Vec<Document>, DataSourceError> {
    let Value::Array(rc) = v else {
        return Err(DataSourceError::Validation {
            message: "aggregation must be an array of pipeline stages".to_string(),
        });
    };
    let mut out = Vec::new();
    for stage in rc.borrow().iter() {
        out.push(value_to_document(stage)?);
    }
    Ok(out)
}

fn value_to_document(v: &Value) -> Result<Document, DataSourceError> {
    match value_to_bson(v) {
        Bson::Document(d) => Ok(d),
        other => Err(DataSourceError::Validation {
            message: format!("expected document object, got {:?}", other.element_type()),
        }),
    }
}

fn json_to_document(v: &serde_json::Value) -> Result<Document, DataSourceError> {
    bson::to_document(v).map_err(|e| DataSourceError::Parse {
        message: format!("MongoSerializationError: {}", e),
    })
}

fn value_to_bson(v: &Value) -> Bson {
    match v {
        Value::Null => Bson::Null,
        Value::Bool(b) => Bson::Boolean(*b),
        Value::Number(n) => {
            if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                Bson::Int64(*n as i64)
            } else {
                Bson::Double(*n)
            }
        }
        Value::Float(f) => match f {
            crate::common::numeric::FloatValue::Finite(n) => Bson::Double(*n),
            _ => Bson::Null,
        },
        Value::Int(IntValue::Finite(n)) => Bson::Int64(*n),
        Value::Int(_) => Bson::Null,
        Value::String(s) => {
            // Try ObjectId hex (24 chars).
            if s.len() == 24 {
                if let Ok(oid) = bson::oid::ObjectId::parse_str(s) {
                    return Bson::ObjectId(oid);
                }
            }
            Bson::String(s.clone())
        }
        Value::Date(dt) => Bson::DateTime(bson::DateTime::from_millis(dt.timestamp_millis())),
        Value::ByteBuffer(b) => {
            let bytes = b.bytes[b.offset..b.offset + b.len].to_vec();
            Bson::Binary(bson::Binary {
                subtype: bson::spec::BinarySubtype::Generic,
                bytes,
            })
        }
        Value::Array(rc) => Bson::Array(rc.borrow().iter().map(value_to_bson).collect()),
        Value::Object(rc) => {
            let kind = rc.borrow();
            let mut doc = Document::new();
            match &*kind {
                ObjectKind::Legacy(m) => {
                    for (k, val) in m {
                        doc.insert(k.clone(), value_to_bson(val));
                    }
                }
                ObjectKind::Inline(entries) => {
                    for (k, val) in entries {
                        if let Value::String(sk) = k {
                            doc.insert(sk.clone(), value_to_bson(val));
                        }
                    }
                }
                ObjectKind::Bucket(_) => {}
            }
            Bson::Document(doc)
        }
        other => Bson::String(other.to_string()),
    }
}

fn bson_document_to_value(doc: &Document) -> Value {
    let mut m = HashMap::new();
    for (k, v) in doc {
        m.insert(k.clone(), bson_to_value(v));
    }
    Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(m))))
}

fn bson_to_value(b: &Bson) -> Value {
    match b {
        Bson::Double(n) => Value::Number(*n),
        Bson::String(s) => Value::String(s.clone()),
        Bson::Array(arr) => {
            Value::Array(Rc::new(RefCell::new(arr.iter().map(bson_to_value).collect())))
        }
        Bson::Document(d) => bson_document_to_value(d),
        Bson::Boolean(b) => Value::Bool(*b),
        Bson::Null => Value::Null,
        Bson::Int32(i) => Value::Int(IntValue::Finite(*i as i64)),
        Bson::Int64(i) => Value::Int(IntValue::Finite(*i)),
        Bson::ObjectId(oid) => Value::String(oid.to_hex()),
        Bson::DateTime(dt) => {
            let millis = dt.timestamp_millis();
            let utc = Utc.timestamp_millis_opt(millis).single().unwrap_or_else(Utc::now);
            Value::Date(utc.with_timezone(&FixedOffset::east_opt(0).unwrap()))
        }
        Bson::Binary(bin) => Value::ByteBuffer(ByteBuffer::from_vec(bin.bytes.clone())),
        Bson::Decimal128(d) => Value::String(d.to_string()),
        Bson::Timestamp(ts) => Value::Number(ts.time as f64),
        Bson::RegularExpression(re) => Value::String(format!("/{}/{}", re.pattern, re.options)),
        Bson::JavaScriptCode(s) | Bson::Symbol(s) => Value::String(s.clone()),
        Bson::JavaScriptCodeWithScope(code) => Value::String(code.code.clone()),
        Bson::MinKey => Value::String("MinKey".into()),
        Bson::MaxKey => Value::String("MaxKey".into()),
        Bson::Undefined => Value::Null,
        Bson::DbPointer(_) => Value::String("DbPointer".into()),
    }
}

// Re-export bson doc! for ping — use bson::doc
use bson::doc;
