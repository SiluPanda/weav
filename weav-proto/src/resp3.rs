//! RESP3 (Redis Serialization Protocol v3) codec implementation.
//!
//! Provides encode/decode of RESP3 frames using `tokio_util::codec`.

use std::io;
use std::str;

use bytes::{Buf, BufMut, BytesMut};
use tokio_util::codec::{Decoder, Encoder};

// ---- RESP3 Value type -------------------------------------------------------

/// A single RESP3 protocol value.
#[derive(Debug, Clone, PartialEq)]
pub enum Resp3Value {
    /// Simple string: `+OK\r\n`
    SimpleString(String),
    /// Blob string: `$<len>\r\n<data>\r\n`
    BlobString(Vec<u8>),
    /// Simple error: `-ERR message\r\n`
    SimpleError(String),
    /// Number: `:<number>\r\n`
    Number(i64),
    /// Double: `,<double>\r\n`
    Double(f64),
    /// Boolean: `#t\r\n` or `#f\r\n`
    Boolean(bool),
    /// Null: `_\r\n`
    Null,
    /// Array: `*<count>\r\n<elements...>`
    Array(Vec<Resp3Value>),
    /// Map: `%<count>\r\n<key><value>...` (count = number of entries, not items)
    Map(Vec<(Resp3Value, Resp3Value)>),
}

// ---- Convenience constructors -----------------------------------------------

impl Resp3Value {
    /// Create a `+OK\r\n` simple string.
    pub fn ok() -> Self {
        Resp3Value::SimpleString("OK".to_string())
    }

    /// Create a simple error.
    pub fn error(msg: impl Into<String>) -> Self {
        Resp3Value::SimpleError(msg.into())
    }

    /// Create a blob string.
    pub fn bulk_string(data: impl Into<Vec<u8>>) -> Self {
        Resp3Value::BlobString(data.into())
    }

    /// Create a number.
    pub fn integer(n: i64) -> Self {
        Resp3Value::Number(n)
    }

    /// Try to interpret this value as a UTF-8 string reference.
    /// Works for both `SimpleString` and `BlobString` (if valid UTF-8).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Resp3Value::SimpleString(s) => Some(s.as_str()),
            Resp3Value::BlobString(b) => str::from_utf8(b).ok(),
            _ => None,
        }
    }

    /// Try to get a byte-slice reference. Works for `BlobString`.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Resp3Value::BlobString(b) => Some(b.as_slice()),
            _ => None,
        }
    }

    /// Consume this value and return the inner array, if it is one.
    pub fn into_array(self) -> Option<Vec<Resp3Value>> {
        match self {
            Resp3Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

// ---- Codec ------------------------------------------------------------------

const DEFAULT_MAX_FRAME_SIZE: usize = 64 * 1024 * 1024; // 64 MB

/// Tokio codec for RESP3 encode/decode.
pub struct Resp3Codec {
    max_frame_size: usize,
}

impl Resp3Codec {
    /// Create a codec with the default 64 MB max frame size.
    pub fn new() -> Self {
        Self {
            max_frame_size: DEFAULT_MAX_FRAME_SIZE,
        }
    }

    /// Create a codec with a custom max frame size (in bytes).
    pub fn with_max_frame_size(max_bytes: usize) -> Self {
        Self {
            max_frame_size: max_bytes,
        }
    }
}

impl Default for Resp3Codec {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Decoder ----------------------------------------------------------------

impl Decoder for Resp3Codec {
    type Item = Resp3Value;
    type Error = io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if src.is_empty() {
            return Ok(None);
        }
        let mut pos = 0;
        match decode_value(src, &mut pos, self.max_frame_size) {
            Ok(Some(val)) => {
                src.advance(pos);
                Ok(Some(val))
            }
            Ok(None) => {
                // Not enough data yet -- reserve a bit more space.
                src.reserve(256);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}

/// Try to decode one RESP3 value starting at `pos` in `buf`.
/// Returns `Ok(None)` when there is not enough data yet.
/// On success, `pos` is advanced past the consumed bytes.
fn decode_value(
    buf: &[u8],
    pos: &mut usize,
    max_size: usize,
) -> Result<Option<Resp3Value>, io::Error> {
    if *pos >= buf.len() {
        return Ok(None);
    }

    let prefix = buf[*pos];
    *pos += 1;

    match prefix {
        b'+' => {
            // SimpleString: read until \r\n
            match read_line(buf, pos)? {
                Some(line) => Ok(Some(Resp3Value::SimpleString(line))),
                None => {
                    *pos -= 1; // rewind prefix byte
                    Ok(None)
                }
            }
        }
        b'-' => {
            // SimpleError: read until \r\n
            match read_line(buf, pos)? {
                Some(line) => Ok(Some(Resp3Value::SimpleError(line))),
                None => {
                    *pos -= 1;
                    Ok(None)
                }
            }
        }
        b':' => {
            // Number: read until \r\n, parse i64
            match read_line(buf, pos)? {
                Some(line) => {
                    let n: i64 = line.parse().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, format!("invalid number: {line}"))
                    })?;
                    Ok(Some(Resp3Value::Number(n)))
                }
                None => {
                    *pos -= 1;
                    Ok(None)
                }
            }
        }
        b',' => {
            // Double: read until \r\n, parse f64
            match read_line(buf, pos)? {
                Some(line) => {
                    let f: f64 = line.parse().map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("invalid double: {line}"),
                        )
                    })?;
                    Ok(Some(Resp3Value::Double(f)))
                }
                None => {
                    *pos -= 1;
                    Ok(None)
                }
            }
        }
        b'#' => {
            // Boolean: next byte is 't' or 'f', then \r\n
            if *pos >= buf.len() {
                *pos -= 1;
                return Ok(None);
            }
            let flag_byte = buf[*pos];
            *pos += 1;
            // Need \r\n
            if *pos + 1 >= buf.len() {
                *pos -= 2; // rewind prefix + flag
                return Ok(None);
            }
            if buf[*pos] != b'\r' || buf[*pos + 1] != b'\n' {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected \\r\\n after boolean flag",
                ));
            }
            *pos += 2;
            match flag_byte {
                b't' => Ok(Some(Resp3Value::Boolean(true))),
                b'f' => Ok(Some(Resp3Value::Boolean(false))),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid boolean flag: {}", flag_byte as char),
                )),
            }
        }
        b'_' => {
            // Null: just \r\n
            if *pos + 1 >= buf.len() {
                *pos -= 1;
                return Ok(None);
            }
            if buf[*pos] != b'\r' || buf[*pos + 1] != b'\n' {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected \\r\\n after null prefix",
                ));
            }
            *pos += 2;
            Ok(Some(Resp3Value::Null))
        }
        b'$' => {
            // BlobString: $<len>\r\n<data>\r\n
            let start = *pos;
            match read_line(buf, pos)? {
                Some(len_str) => {
                    let len: usize = len_str.parse().map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("invalid blob length: {len_str}"),
                        )
                    })?;
                    if len > max_size {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("blob string length {len} exceeds max frame size {max_size}"),
                        ));
                    }
                    // Need `len` bytes of data + \r\n
                    if *pos + len + 2 > buf.len() {
                        // Not enough data yet -- rewind everything.
                        *pos = start - 1; // rewind to before '$'
                        return Ok(None);
                    }
                    let data = buf[*pos..*pos + len].to_vec();
                    *pos += len;
                    if buf[*pos] != b'\r' || buf[*pos + 1] != b'\n' {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "missing \\r\\n after blob data",
                        ));
                    }
                    *pos += 2;
                    Ok(Some(Resp3Value::BlobString(data)))
                }
                None => {
                    *pos = start - 1;
                    Ok(None)
                }
            }
        }
        b'*' => {
            // Array: *<count>\r\n then count elements
            let start = *pos;
            match read_line(buf, pos)? {
                Some(count_str) => {
                    let count: usize = count_str.parse().map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("invalid array count: {count_str}"),
                        )
                    })?;
                    let mut items = Vec::with_capacity(count.min(1024));
                    for _ in 0..count {
                        match decode_value(buf, pos, max_size)? {
                            Some(val) => items.push(val),
                            None => {
                                *pos = start - 1;
                                return Ok(None);
                            }
                        }
                    }
                    Ok(Some(Resp3Value::Array(items)))
                }
                None => {
                    *pos = start - 1;
                    Ok(None)
                }
            }
        }
        b'%' => {
            // Map: %<count>\r\n then count*2 elements (key,value pairs)
            let start = *pos;
            match read_line(buf, pos)? {
                Some(count_str) => {
                    let count: usize = count_str.parse().map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("invalid map count: {count_str}"),
                        )
                    })?;
                    let mut pairs = Vec::with_capacity(count.min(1024));
                    for _ in 0..count {
                        let key = match decode_value(buf, pos, max_size)? {
                            Some(v) => v,
                            None => {
                                *pos = start - 1;
                                return Ok(None);
                            }
                        };
                        let val = match decode_value(buf, pos, max_size)? {
                            Some(v) => v,
                            None => {
                                *pos = start - 1;
                                return Ok(None);
                            }
                        };
                        pairs.push((key, val));
                    }
                    Ok(Some(Resp3Value::Map(pairs)))
                }
                None => {
                    *pos = start - 1;
                    Ok(None)
                }
            }
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown RESP3 type prefix: {:?}", other as char),
        )),
    }
}

/// Read until `\r\n` starting at `*pos`.
/// Returns `Ok(Some(line_content))` with `pos` advanced past `\r\n`,
/// or `Ok(None)` if the terminator hasn't arrived yet.
fn read_line(buf: &[u8], pos: &mut usize) -> Result<Option<String>, io::Error> {
    let start = *pos;
    while *pos < buf.len() {
        if buf[*pos] == b'\r' {
            if *pos + 1 < buf.len() {
                if buf[*pos + 1] == b'\n' {
                    let line = &buf[start..*pos];
                    let s = str::from_utf8(line).map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {e}"))
                    })?;
                    *pos += 2; // skip \r\n
                    return Ok(Some(s.to_string()));
                } else {
                    // \r not followed by \n -- protocol error
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "\\r not followed by \\n",
                    ));
                }
            } else {
                // We see \r at the very end -- might be partial \r\n
                *pos = start;
                return Ok(None);
            }
        }
        *pos += 1;
    }
    // Reached end of buffer without finding \r\n
    *pos = start;
    Ok(None)
}

// ---- Encoder ----------------------------------------------------------------

impl Encoder<Resp3Value> for Resp3Codec {
    type Error = io::Error;

    fn encode(&mut self, item: Resp3Value, dst: &mut BytesMut) -> Result<(), Self::Error> {
        encode_value(&item, dst);
        Ok(())
    }
}

/// Encode a single `Resp3Value` into the buffer.
fn encode_value(val: &Resp3Value, dst: &mut BytesMut) {
    match val {
        Resp3Value::SimpleString(s) => {
            dst.put_u8(b'+');
            dst.put_slice(s.as_bytes());
            dst.put_slice(b"\r\n");
        }
        Resp3Value::SimpleError(s) => {
            dst.put_u8(b'-');
            dst.put_slice(s.as_bytes());
            dst.put_slice(b"\r\n");
        }
        Resp3Value::Number(n) => {
            dst.put_u8(b':');
            dst.put_slice(n.to_string().as_bytes());
            dst.put_slice(b"\r\n");
        }
        Resp3Value::Double(f) => {
            dst.put_u8(b',');
            // Use a representation that round-trips losslessly.
            let s = format!("{f}");
            dst.put_slice(s.as_bytes());
            dst.put_slice(b"\r\n");
        }
        Resp3Value::Boolean(b) => {
            dst.put_u8(b'#');
            dst.put_u8(if *b { b't' } else { b'f' });
            dst.put_slice(b"\r\n");
        }
        Resp3Value::Null => {
            dst.put_slice(b"_\r\n");
        }
        Resp3Value::BlobString(data) => {
            dst.put_u8(b'$');
            dst.put_slice(data.len().to_string().as_bytes());
            dst.put_slice(b"\r\n");
            dst.put_slice(data);
            dst.put_slice(b"\r\n");
        }
        Resp3Value::Array(items) => {
            dst.put_u8(b'*');
            dst.put_slice(items.len().to_string().as_bytes());
            dst.put_slice(b"\r\n");
            for item in items {
                encode_value(item, dst);
            }
        }
        Resp3Value::Map(pairs) => {
            dst.put_u8(b'%');
            dst.put_slice(pairs.len().to_string().as_bytes());
            dst.put_slice(b"\r\n");
            for (k, v) in pairs {
                encode_value(k, dst);
                encode_value(v, dst);
            }
        }
    }
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: encode a value then decode it back and assert round-trip equality.
    fn roundtrip(val: Resp3Value) {
        let mut codec = Resp3Codec::new();
        let mut buf = BytesMut::new();
        codec.encode(val.clone(), &mut buf).unwrap();
        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(val, decoded);
    }

    #[test]
    fn test_simple_string_roundtrip() {
        roundtrip(Resp3Value::SimpleString("OK".to_string()));
        roundtrip(Resp3Value::SimpleString("hello world".to_string()));
        roundtrip(Resp3Value::SimpleString(String::new()));
    }

    #[test]
    fn test_simple_error_roundtrip() {
        roundtrip(Resp3Value::SimpleError("ERR something broke".to_string()));
    }

    #[test]
    fn test_number_roundtrip() {
        roundtrip(Resp3Value::Number(0));
        roundtrip(Resp3Value::Number(42));
        roundtrip(Resp3Value::Number(-100));
        roundtrip(Resp3Value::Number(i64::MAX));
        roundtrip(Resp3Value::Number(i64::MIN));
    }

    #[test]
    fn test_double_roundtrip() {
        roundtrip(Resp3Value::Double(0.0));
        roundtrip(Resp3Value::Double(3.14));
        roundtrip(Resp3Value::Double(-2.718));
        roundtrip(Resp3Value::Double(1.0));
    }

    #[test]
    fn test_boolean_roundtrip() {
        roundtrip(Resp3Value::Boolean(true));
        roundtrip(Resp3Value::Boolean(false));
    }

    #[test]
    fn test_null_roundtrip() {
        roundtrip(Resp3Value::Null);
    }

    #[test]
    fn test_blob_string_roundtrip() {
        roundtrip(Resp3Value::BlobString(b"hello".to_vec()));
        roundtrip(Resp3Value::BlobString(b"".to_vec()));
        roundtrip(Resp3Value::BlobString(vec![0, 1, 2, 255, 254]));
        // Blob with \r\n embedded
        roundtrip(Resp3Value::BlobString(b"line1\r\nline2".to_vec()));
    }

    #[test]
    fn test_array_roundtrip() {
        roundtrip(Resp3Value::Array(vec![]));
        roundtrip(Resp3Value::Array(vec![
            Resp3Value::SimpleString("hello".to_string()),
            Resp3Value::Number(42),
            Resp3Value::Null,
        ]));
    }

    #[test]
    fn test_map_roundtrip() {
        roundtrip(Resp3Value::Map(vec![]));
        roundtrip(Resp3Value::Map(vec![
            (
                Resp3Value::SimpleString("key1".to_string()),
                Resp3Value::Number(1),
            ),
            (
                Resp3Value::SimpleString("key2".to_string()),
                Resp3Value::Boolean(true),
            ),
        ]));
    }

    #[test]
    fn test_nested_array_roundtrip() {
        let val = Resp3Value::Array(vec![
            Resp3Value::Array(vec![
                Resp3Value::Number(1),
                Resp3Value::Number(2),
            ]),
            Resp3Value::Array(vec![
                Resp3Value::SimpleString("a".to_string()),
                Resp3Value::BlobString(b"b".to_vec()),
            ]),
        ]);
        roundtrip(val);
    }

    #[test]
    fn test_nested_map_in_array() {
        let val = Resp3Value::Array(vec![
            Resp3Value::Map(vec![(
                Resp3Value::SimpleString("inner".to_string()),
                Resp3Value::Double(9.99),
            )]),
            Resp3Value::Null,
        ]);
        roundtrip(val);
    }

    #[test]
    fn test_deeply_nested() {
        let val = Resp3Value::Map(vec![(
            Resp3Value::SimpleString("level1".to_string()),
            Resp3Value::Map(vec![(
                Resp3Value::SimpleString("level2".to_string()),
                Resp3Value::Array(vec![
                    Resp3Value::Number(1),
                    Resp3Value::Map(vec![(
                        Resp3Value::SimpleString("level3".to_string()),
                        Resp3Value::Boolean(true),
                    )]),
                ]),
            )]),
        )]);
        roundtrip(val);
    }

    #[test]
    fn test_partial_frame_returns_none() {
        let mut codec = Resp3Codec::new();

        // Only send the prefix byte for a simple string, no \r\n yet.
        let mut buf = BytesMut::from(&b"+hel"[..]);
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_none());
        // Buffer should not be consumed.
        assert_eq!(&buf[..], b"+hel");

        // Now complete the frame.
        buf.put_slice(b"lo\r\n");
        let result = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(result, Resp3Value::SimpleString("hello".to_string()));
        assert!(buf.is_empty());
    }

    #[test]
    fn test_partial_blob_string() {
        let mut codec = Resp3Codec::new();

        // Send length but not the data.
        let mut buf = BytesMut::from(&b"$5\r\nhel"[..]);
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_none());

        // Complete the data.
        buf.put_slice(b"lo\r\n");
        let result = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(result, Resp3Value::BlobString(b"hello".to_vec()));
    }

    #[test]
    fn test_partial_array() {
        let mut codec = Resp3Codec::new();

        // Array header and first element, but missing the second.
        let mut buf = BytesMut::from(&b"*2\r\n:1\r\n"[..]);
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_none());

        // Complete with second element.
        buf.put_slice(b":2\r\n");
        let result = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(
            result,
            Resp3Value::Array(vec![Resp3Value::Number(1), Resp3Value::Number(2)])
        );
    }

    #[test]
    fn test_partial_null() {
        let mut codec = Resp3Codec::new();

        let mut buf = BytesMut::from(&b"_"[..]);
        assert!(codec.decode(&mut buf).unwrap().is_none());

        buf.put_slice(b"\r\n");
        let result = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(result, Resp3Value::Null);
    }

    #[test]
    fn test_partial_boolean() {
        let mut codec = Resp3Codec::new();

        // Just the prefix
        let mut buf = BytesMut::from(&b"#"[..]);
        assert!(codec.decode(&mut buf).unwrap().is_none());

        buf.put_slice(b"t\r\n");
        let result = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(result, Resp3Value::Boolean(true));
    }

    #[test]
    fn test_multiple_frames_in_buffer() {
        let mut codec = Resp3Codec::new();
        let mut buf = BytesMut::new();

        // Put two frames into the buffer.
        codec
            .encode(Resp3Value::SimpleString("first".to_string()), &mut buf)
            .unwrap();
        codec
            .encode(Resp3Value::Number(42), &mut buf)
            .unwrap();

        let v1 = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(v1, Resp3Value::SimpleString("first".to_string()));

        let v2 = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(v2, Resp3Value::Number(42));

        // Nothing left.
        assert!(codec.decode(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_unknown_prefix_is_error() {
        let mut codec = Resp3Codec::new();
        let mut buf = BytesMut::from(&b"X\r\n"[..]);
        let result = codec.decode(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_convenience_constructors() {
        assert_eq!(Resp3Value::ok(), Resp3Value::SimpleString("OK".to_string()));
        assert_eq!(
            Resp3Value::error("ERR fail"),
            Resp3Value::SimpleError("ERR fail".to_string())
        );
        assert_eq!(
            Resp3Value::bulk_string(b"data".to_vec()),
            Resp3Value::BlobString(b"data".to_vec())
        );
        assert_eq!(Resp3Value::integer(99), Resp3Value::Number(99));
    }

    #[test]
    fn test_as_str() {
        let ss = Resp3Value::SimpleString("hello".to_string());
        assert_eq!(ss.as_str(), Some("hello"));

        let bs = Resp3Value::BlobString(b"world".to_vec());
        assert_eq!(bs.as_str(), Some("world"));

        let bad = Resp3Value::BlobString(vec![0xff, 0xfe]);
        assert!(bad.as_str().is_none());

        let num = Resp3Value::Number(42);
        assert!(num.as_str().is_none());
    }

    #[test]
    fn test_as_bytes() {
        let bs = Resp3Value::BlobString(b"hello".to_vec());
        assert_eq!(bs.as_bytes(), Some(&b"hello"[..]));

        let ss = Resp3Value::SimpleString("hello".to_string());
        assert!(ss.as_bytes().is_none());
    }

    #[test]
    fn test_into_array() {
        let arr = Resp3Value::Array(vec![Resp3Value::Number(1)]);
        let inner = arr.into_array().unwrap();
        assert_eq!(inner.len(), 1);

        let non_arr = Resp3Value::Null;
        assert!(non_arr.into_array().is_none());
    }

    #[test]
    fn test_max_frame_size_blob() {
        let mut codec = Resp3Codec::with_max_frame_size(10);
        // Try to decode a blob that claims to be 100 bytes.
        let mut buf = BytesMut::from(&b"$100\r\n"[..]);
        // Pad with enough data.
        buf.put_bytes(b'A', 100);
        buf.put_slice(b"\r\n");
        let result = codec.decode(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_double_precision() {
        // Make sure doubles encode in a way that round-trips.
        roundtrip(Resp3Value::Double(1.0));
        roundtrip(Resp3Value::Double(0.1));
        roundtrip(Resp3Value::Double(-0.0));
    }

    #[test]
    fn test_large_blob() {
        let data = vec![42u8; 100_000];
        roundtrip(Resp3Value::BlobString(data));
    }

    #[test]
    fn test_partial_map() {
        let mut codec = Resp3Codec::new();

        // Map header saying 1 entry, but only key present, no value yet.
        let mut buf = BytesMut::from(&b"%1\r\n+key\r\n"[..]);
        assert!(codec.decode(&mut buf).unwrap().is_none());

        // Complete the value.
        buf.put_slice(b":42\r\n");
        let result = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(
            result,
            Resp3Value::Map(vec![(
                Resp3Value::SimpleString("key".to_string()),
                Resp3Value::Number(42),
            )])
        );
    }

    #[test]
    fn test_empty_simple_string() {
        roundtrip(Resp3Value::SimpleString(String::new()));
    }

    #[test]
    fn test_empty_blob_string() {
        roundtrip(Resp3Value::BlobString(vec![]));
    }
}
