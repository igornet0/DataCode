//! Host/runtime value payloads referenced by [`crate::common::value::Value`].
//! Keeps `value.rs` focused on VM semantics; plot/database types are centralized here for future
//! migration into a single opaque layer.

pub(crate) use crate::database_engine::cluster::DatabaseCluster;
pub(crate) use crate::database_engine::engine::DatabaseEngine;
pub(crate) use crate::plot::{Axis, Figure, Image, PlotWindowHandle};
