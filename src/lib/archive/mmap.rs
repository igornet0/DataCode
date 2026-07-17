//! Memory-mapped archive file source.

use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug)]
pub enum ArchiveData {
    Mmap(Arc<Mmap>),
    Vec(Vec<u8>),
}

impl ArchiveData {
    pub fn len(&self) -> usize {
        match self {
            ArchiveData::Mmap(m) => m.len(),
            ArchiveData::Vec(v) => v.len(),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            ArchiveData::Mmap(m) => m.as_ref(),
            ArchiveData::Vec(v) => v.as_ref(),
        }
    }

    pub fn cursor(&self) -> ArchiveCursor {
        ArchiveCursor {
            data: self.clone_inner(),
            pos: 0,
        }
    }

    fn clone_inner(&self) -> ArchiveData {
        match self {
            ArchiveData::Mmap(m) => ArchiveData::Mmap(Arc::clone(m)),
            ArchiveData::Vec(v) => ArchiveData::Vec(v.clone()),
        }
    }
}

#[derive(Debug)]
pub struct ArchiveCursor {
    data: ArchiveData,
    pos: u64,
}

impl Read for ArchiveCursor {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let slice = self.data.as_slice();
        let start = self.pos as usize;
        if start >= slice.len() {
            return Ok(0);
        }
        let end = (start + buf.len()).min(slice.len());
        let n = end - start;
        buf[..n].copy_from_slice(&slice[start..end]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for ArchiveCursor {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let len = self.data.len() as u64;
        let new_pos = match pos {
            SeekFrom::Start(p) => p,
            SeekFrom::End(off) => {
                if off >= 0 {
                    len.saturating_add(off as u64)
                } else {
                    len.saturating_sub((-off) as u64)
                }
            }
            SeekFrom::Current(off) => {
                if off >= 0 {
                    self.pos.saturating_add(off as u64)
                } else {
                    self.pos.saturating_sub((-off) as u64)
                }
            }
        };
        self.pos = new_pos.min(len);
        Ok(self.pos)
    }
}

#[derive(Debug)]
pub struct ArchiveSource {
    pub path: PathBuf,
    pub data: ArchiveData,
    pub file_size: u64,
}

impl ArchiveSource {
    pub fn open(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("Archive not found: '{}'", path.display()));
        }
        let mut file = File::open(path).map_err(|e| {
            format!("Archive not found: '{}': {}", path.display(), e)
        })?;
        let file_size = file.metadata().map_err(|e| e.to_string())?.len();
        let data = match unsafe { Mmap::map(&file) } {
            Ok(mmap) => ArchiveData::Mmap(Arc::new(mmap)),
            Err(_) => {
                let mut buf = Vec::with_capacity(file_size as usize);
                file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
                ArchiveData::Vec(buf)
            }
        };
        Ok(Self {
            path: path.to_path_buf(),
            data,
            file_size,
        })
    }

    pub fn cursor(&self) -> ArchiveCursor {
        self.data.cursor()
    }
}
