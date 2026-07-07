#![deny(unsafe_op_in_unsafe_fn)]

//! File IO helpers for large GLM weight shards.

#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::{
    fs::File,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
};

use common::{Error, Result};
#[cfg(unix)]
use memmap2::Advice;
use memmap2::{Mmap, MmapOptions};

const PREFETCH_CHUNK_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappedFileAdvice {
    Random,
    WillNeed,
}

#[derive(Debug)]
pub struct MappedFile {
    path: PathBuf,
    file: File,
    mmap: Mmap,
}

impl MappedFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let len = file
            .metadata()
            .map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?
            .len();
        if len == 0 {
            return Err(Error::weights(format!(
                "cannot memory-map empty file {}",
                path.display()
            )));
        }

        // SAFETY: The map is read-only and the returned Mmap owns the mapping.
        // Callers only receive bounds-checked shared byte slices from this wrapper.
        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?
        };
        Ok(Self {
            path: path.to_path_buf(),
            file,
            mmap,
        })
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn cache_identity(&self) -> usize {
        self.mmap.as_ptr() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    pub fn slice(&self, offset: u64, byte_len: usize) -> Result<&[u8]> {
        let offset = usize::try_from(offset).map_err(|_| {
            Error::weights(format!(
                "mapped file offset does not fit usize for {}",
                self.path.display()
            ))
        })?;
        let end = offset.checked_add(byte_len).ok_or_else(|| {
            Error::weights(format!(
                "mapped file slice range overflows for {}",
                self.path.display()
            ))
        })?;
        if end > self.mmap.len() {
            return Err(Error::weights(format!(
                "mapped file slice [{}..{}] exceeds file size {} for {}",
                offset,
                end,
                self.mmap.len(),
                self.path.display()
            )));
        }
        Ok(&self.mmap[offset..end])
    }

    pub fn advise_range(&self, advice: MappedFileAdvice, offset: u64, byte_len: u64) -> Result<()> {
        if byte_len == 0 {
            return Ok(());
        }
        let offset = usize::try_from(offset).map_err(|_| {
            Error::weights(format!(
                "mapped file advice offset does not fit usize for {}",
                self.path.display()
            ))
        })?;
        let byte_len = usize::try_from(byte_len).map_err(|_| {
            Error::weights(format!(
                "mapped file advice length does not fit usize for {}",
                self.path.display()
            ))
        })?;
        let end = offset.checked_add(byte_len).ok_or_else(|| {
            Error::weights(format!(
                "mapped file advice range overflows for {}",
                self.path.display()
            ))
        })?;
        if end > self.mmap.len() {
            return Err(Error::weights(format!(
                "mapped file advice range [{}..{}] exceeds file size {} for {}",
                offset,
                end,
                self.mmap.len(),
                self.path.display()
            )));
        }

        #[cfg(unix)]
        self.mmap
            .advise_range(advice.into(), offset, byte_len)
            .map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;

        #[cfg(not(unix))]
        let _ = advice;

        Ok(())
    }

    pub fn prefetch_range(&self, offset: u64, byte_len: u64) -> Result<()> {
        self.prefetch_ranges(&[(offset, byte_len)])
    }

    pub fn prefetch_ranges(&self, ranges: &[(u64, u64)]) -> Result<()> {
        if ranges.is_empty() {
            return Ok(());
        }

        for &(offset, byte_len) in ranges {
            if byte_len == 0 {
                continue;
            }
            self.validate_range("prefetch", offset, byte_len)?;
        }

        for &(offset, byte_len) in ranges {
            if byte_len == 0 {
                continue;
            }
            self.advise_range(MappedFileAdvice::WillNeed, offset, byte_len)?;
        }

        #[cfg(unix)]
        {
            let max_range_len = ranges
                .iter()
                .map(|&(_, byte_len)| byte_len)
                .max()
                .unwrap_or(0);
            if max_range_len == 0 {
                return Ok(());
            }
            let buffer_len = usize::try_from(max_range_len)
                .ok()
                .map(|len| len.min(PREFETCH_CHUNK_BYTES))
                .unwrap_or(PREFETCH_CHUNK_BYTES);
            let mut buffer = vec![0_u8; buffer_len];

            for &(offset, byte_len) in ranges {
                self.prefetch_range_with_buffer(offset, byte_len, &mut buffer)?;
            }
        }

        #[cfg(not(unix))]
        let _ = ranges;

        Ok(())
    }

    #[cfg(unix)]
    fn prefetch_range_with_buffer(
        &self,
        offset: u64,
        byte_len: u64,
        buffer: &mut [u8],
    ) -> Result<()> {
        if byte_len == 0 {
            return Ok(());
        }

        let mut remaining = byte_len;
        let mut read_offset = offset;
        while remaining > 0 {
            let chunk_len = usize::try_from(remaining)
                .ok()
                .map(|len| len.min(buffer.len()))
                .unwrap_or(buffer.len());
            self.read_exact_at(&mut buffer[..chunk_len], read_offset)?;
            read_offset = read_offset
                .checked_add(u64::try_from(chunk_len).map_err(|_| {
                    Error::weights("mapped file prefetch chunk length does not fit u64")
                })?)
                .ok_or_else(|| Error::weights("mapped file prefetch offset overflow"))?;
            remaining -= u64::try_from(chunk_len).map_err(|_| {
                Error::weights("mapped file prefetch chunk length does not fit u64")
            })?;
        }

        Ok(())
    }

    fn validate_range(&self, label: &str, offset: u64, byte_len: u64) -> Result<()> {
        let offset = usize::try_from(offset).map_err(|_| {
            Error::weights(format!(
                "mapped file {label} offset does not fit usize for {}",
                self.path.display()
            ))
        })?;
        let byte_len = usize::try_from(byte_len).map_err(|_| {
            Error::weights(format!(
                "mapped file {label} length does not fit usize for {}",
                self.path.display()
            ))
        })?;
        let end = offset.checked_add(byte_len).ok_or_else(|| {
            Error::weights(format!(
                "mapped file {label} range overflows for {}",
                self.path.display()
            ))
        })?;
        if end > self.mmap.len() {
            return Err(Error::weights(format!(
                "mapped file {label} range [{}..{}] exceeds file size {} for {}",
                offset,
                end,
                self.mmap.len(),
                self.path.display()
            )));
        }
        Ok(())
    }

    #[cfg(unix)]
    fn read_exact_at(&self, mut buffer: &mut [u8], mut offset: u64) -> Result<()> {
        while !buffer.is_empty() {
            let read = self
                .file
                .read_at(buffer, offset)
                .map_err(|source| Error::Io {
                    path: self.path.clone(),
                    source,
                })?;
            if read == 0 {
                return Err(Error::Io {
                    path: self.path.clone(),
                    source: io::Error::new(
                        ErrorKind::UnexpectedEof,
                        "unexpected EOF while prefetching mapped file range",
                    ),
                });
            }
            offset =
                offset
                    .checked_add(u64::try_from(read).map_err(|_| {
                        Error::weights("mapped file read byte count does not fit u64")
                    })?)
                    .ok_or_else(|| Error::weights("mapped file read offset overflow"))?;
            buffer = &mut buffer[read..];
        }
        Ok(())
    }
}

#[cfg(unix)]
impl From<MappedFileAdvice> for Advice {
    fn from(value: MappedFileAdvice) -> Self {
        match value {
            MappedFileAdvice::Random => Self::Random,
            MappedFileAdvice::WillNeed => Self::WillNeed,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn mapped_file_returns_checked_slice() {
        let path = unique_temp_file("slice");
        fs::write(&path, b"abcdef").unwrap();

        let mapped = MappedFile::open(&path).unwrap();

        assert_eq!(mapped.len(), 6);
        assert_eq!(mapped.slice(2, 3).unwrap(), b"cde");
    }

    #[test]
    fn mapped_file_rejects_out_of_bounds_slice() {
        let path = unique_temp_file("oob");
        fs::write(&path, b"abcdef").unwrap();

        let mapped = MappedFile::open(&path).unwrap();
        let err = mapped
            .slice(4, 3)
            .expect_err("out-of-bounds slice should fail");

        assert!(err.to_string().contains("exceeds file size"));
    }

    #[test]
    fn mapped_file_rejects_out_of_bounds_advice_range() {
        let path = unique_temp_file("advise-oob");
        fs::write(&path, b"abcdef").unwrap();

        let mapped = MappedFile::open(&path).unwrap();
        let err = mapped
            .advise_range(MappedFileAdvice::Random, 4, 3)
            .expect_err("out-of-bounds advice should fail");

        assert!(err.to_string().contains("exceeds file size"));
    }

    #[test]
    fn mapped_file_prefetches_checked_range() {
        let path = unique_temp_file("prefetch");
        fs::write(&path, b"abcdef").unwrap();

        let mapped = MappedFile::open(&path).unwrap();

        mapped.prefetch_range(1, 4).unwrap();
    }

    #[test]
    fn mapped_file_prefetches_multiple_checked_ranges() {
        let path = unique_temp_file("prefetch-ranges");
        fs::write(&path, b"abcdef").unwrap();

        let mapped = MappedFile::open(&path).unwrap();

        mapped.prefetch_ranges(&[(0, 2), (3, 2)]).unwrap();
    }

    #[test]
    fn mapped_file_rejects_out_of_bounds_prefetch_range() {
        let path = unique_temp_file("prefetch-oob");
        fs::write(&path, b"abcdef").unwrap();

        let mapped = MappedFile::open(&path).unwrap();
        let err = mapped
            .prefetch_range(4, 3)
            .expect_err("out-of-bounds prefetch should fail");

        assert!(err.to_string().contains("exceeds file size"));
    }

    #[test]
    fn mapped_file_rejects_empty_file() {
        let path = unique_temp_file("empty");
        fs::write(&path, []).unwrap();

        let err = MappedFile::open(&path).expect_err("empty mmap should fail");

        assert!(err.to_string().contains("empty file"));
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("io-{label}-{}-{id}", std::process::id()))
    }
}
