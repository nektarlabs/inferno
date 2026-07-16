use common::{Error, Result};

pub const EXPERT_PACK_FILE_NAME: &str = "GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack";
pub const EXPERT_PACK_HEADER_BYTES: usize = 4096;

const MAGIC: [u8; 8] = *b"INFRQ2E\0";
const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertComponent {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPackHeader {
    pub source_bytes: u64,
    pub layout_fingerprint: u64,
    pub first_layer: u32,
    pub layer_count: u32,
    pub expert_count: u32,
    pub gate_bytes: u64,
    pub up_bytes: u64,
    pub down_bytes: u64,
}

impl ExpertPackHeader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_bytes: u64,
        layout_fingerprint: u64,
        first_layer: u32,
        layer_count: u32,
        expert_count: u32,
        gate_bytes: u64,
        up_bytes: u64,
        down_bytes: u64,
    ) -> Result<Self> {
        let header = Self {
            source_bytes,
            layout_fingerprint,
            first_layer,
            layer_count,
            expert_count,
            gate_bytes,
            up_bytes,
            down_bytes,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn record_bytes(self) -> Result<u64> {
        self.gate_bytes
            .checked_add(self.up_bytes)
            .and_then(|bytes| bytes.checked_add(self.down_bytes))
            .ok_or_else(|| Error::weights("Q2 expert-pack record byte count overflow"))
    }

    pub fn expected_file_bytes(self) -> Result<u64> {
        let records = u64::from(self.layer_count)
            .checked_mul(u64::from(self.expert_count))
            .ok_or_else(|| Error::weights("Q2 expert-pack record count overflow"))?;
        let data_bytes = records
            .checked_mul(self.record_bytes()?)
            .ok_or_else(|| Error::weights("Q2 expert-pack data byte count overflow"))?;
        (EXPERT_PACK_HEADER_BYTES as u64)
            .checked_add(data_bytes)
            .ok_or_else(|| Error::weights("Q2 expert-pack file byte count overflow"))
    }

    pub fn component_offset(
        self,
        layer_index: u32,
        expert_id: u32,
        component: ExpertComponent,
    ) -> Result<u64> {
        self.validate()?;
        let relative_layer = layer_index.checked_sub(self.first_layer).ok_or_else(|| {
            Error::weights(format!(
                "Q2 expert-pack layer {layer_index} precedes first layer {}",
                self.first_layer
            ))
        })?;
        if relative_layer >= self.layer_count {
            return Err(Error::weights(format!(
                "Q2 expert-pack layer {layer_index} exceeds last layer {}",
                self.first_layer + self.layer_count - 1
            )));
        }
        if expert_id >= self.expert_count {
            return Err(Error::weights(format!(
                "Q2 expert-pack expert {expert_id} exceeds expert count {}",
                self.expert_count
            )));
        }

        let record_index = u64::from(relative_layer)
            .checked_mul(u64::from(self.expert_count))
            .and_then(|index| index.checked_add(u64::from(expert_id)))
            .ok_or_else(|| Error::weights("Q2 expert-pack record index overflow"))?;
        let record_offset = record_index
            .checked_mul(self.record_bytes()?)
            .and_then(|offset| offset.checked_add(EXPERT_PACK_HEADER_BYTES as u64))
            .ok_or_else(|| Error::weights("Q2 expert-pack record offset overflow"))?;
        let component_offset = match component {
            ExpertComponent::Gate => 0,
            ExpertComponent::Up => self.gate_bytes,
            ExpertComponent::Down => self
                .gate_bytes
                .checked_add(self.up_bytes)
                .ok_or_else(|| Error::weights("Q2 expert-pack down offset overflow"))?,
        };
        record_offset
            .checked_add(component_offset)
            .ok_or_else(|| Error::weights("Q2 expert-pack component offset overflow"))
    }

    pub fn encode(self) -> Result<[u8; EXPERT_PACK_HEADER_BYTES]> {
        self.validate()?;
        let mut bytes = [0_u8; EXPERT_PACK_HEADER_BYTES];
        bytes[..8].copy_from_slice(&MAGIC);
        write_u32(&mut bytes, 8, VERSION);
        write_u32(&mut bytes, 12, EXPERT_PACK_HEADER_BYTES as u32);
        write_u64(&mut bytes, 16, self.source_bytes);
        write_u64(&mut bytes, 24, self.layout_fingerprint);
        write_u32(&mut bytes, 32, self.first_layer);
        write_u32(&mut bytes, 36, self.layer_count);
        write_u32(&mut bytes, 40, self.expert_count);
        write_u64(&mut bytes, 48, self.gate_bytes);
        write_u64(&mut bytes, 56, self.up_bytes);
        write_u64(&mut bytes, 64, self.down_bytes);
        write_u64(&mut bytes, 72, self.expected_file_bytes()?);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < EXPERT_PACK_HEADER_BYTES {
            return Err(Error::weights(format!(
                "Q2 expert-pack header requires {EXPERT_PACK_HEADER_BYTES} bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[..8] != MAGIC {
            return Err(Error::weights("invalid Q2 expert-pack magic"));
        }
        let version = read_u32(bytes, 8)?;
        if version != VERSION {
            return Err(Error::weights(format!(
                "unsupported Q2 expert-pack version {version}; expected {VERSION}"
            )));
        }
        let header_bytes = read_u32(bytes, 12)?;
        if header_bytes != EXPERT_PACK_HEADER_BYTES as u32 {
            return Err(Error::weights(format!(
                "Q2 expert-pack header size is {header_bytes}; expected {EXPERT_PACK_HEADER_BYTES}"
            )));
        }
        let header = Self::new(
            read_u64(bytes, 16)?,
            read_u64(bytes, 24)?,
            read_u32(bytes, 32)?,
            read_u32(bytes, 36)?,
            read_u32(bytes, 40)?,
            read_u64(bytes, 48)?,
            read_u64(bytes, 56)?,
            read_u64(bytes, 64)?,
        )?;
        let encoded_file_bytes = read_u64(bytes, 72)?;
        let expected_file_bytes = header.expected_file_bytes()?;
        if encoded_file_bytes != expected_file_bytes {
            return Err(Error::weights(format!(
                "Q2 expert-pack encoded size {encoded_file_bytes} does not match layout size {expected_file_bytes}"
            )));
        }
        Ok(header)
    }

    pub fn validate_file_bytes(self, actual_file_bytes: u64) -> Result<()> {
        let expected_file_bytes = self.expected_file_bytes()?;
        if actual_file_bytes != expected_file_bytes {
            return Err(Error::weights(format!(
                "Q2 expert-pack file has {actual_file_bytes} bytes; expected {expected_file_bytes}"
            )));
        }
        Ok(())
    }

    fn validate(self) -> Result<()> {
        if self.source_bytes == 0
            || self.layout_fingerprint == 0
            || self.layer_count == 0
            || self.expert_count == 0
            || self.gate_bytes == 0
            || self.up_bytes == 0
            || self.down_bytes == 0
        {
            return Err(Error::weights(
                "Q2 expert-pack dimensions and source size must be positive",
            ));
        }
        self.first_layer
            .checked_add(self.layer_count - 1)
            .ok_or_else(|| Error::weights("Q2 expert-pack layer range overflow"))?;
        self.record_bytes()?;
        Ok(())
    }
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::weights("truncated Q2 expert-pack u32 field"))?;
    Ok(u32::from_le_bytes(raw.try_into().map_err(|_| {
        Error::weights("invalid Q2 expert-pack u32 field")
    })?))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error::weights("truncated Q2 expert-pack u64 field"))?;
    Ok(u64::from_le_bytes(raw.try_into().map_err(|_| {
        Error::weights("invalid Q2 expert-pack u64 field")
    })?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> ExpertPackHeader {
        ExpertPackHeader::new(262_000_000_000, 17, 3, 76, 256, 4, 5, 6).unwrap()
    }

    #[test]
    fn header_round_trips() {
        let header = header();
        let encoded = header.encode().unwrap();

        assert_eq!(ExpertPackHeader::decode(&encoded).unwrap(), header);
        header
            .validate_file_bytes(header.expected_file_bytes().unwrap())
            .unwrap();
    }

    #[test]
    fn component_offsets_use_fixed_layer_expert_records() {
        let header = header();
        let record_bytes = 15_u64;

        assert_eq!(
            header
                .component_offset(3, 0, ExpertComponent::Gate)
                .unwrap(),
            EXPERT_PACK_HEADER_BYTES as u64
        );
        assert_eq!(
            header.component_offset(3, 0, ExpertComponent::Up).unwrap(),
            EXPERT_PACK_HEADER_BYTES as u64 + 4
        );
        assert_eq!(
            header
                .component_offset(4, 2, ExpertComponent::Down)
                .unwrap(),
            EXPERT_PACK_HEADER_BYTES as u64 + (258 * record_bytes) + 9
        );
    }

    #[test]
    fn rejects_wrong_magic_and_truncated_headers() {
        let mut encoded = header().encode().unwrap();
        encoded[0] = b'X';
        assert!(ExpertPackHeader::decode(&encoded)
            .unwrap_err()
            .to_string()
            .contains("magic"));
        assert!(ExpertPackHeader::decode(&encoded[..100])
            .unwrap_err()
            .to_string()
            .contains("requires"));
    }

    #[test]
    fn rejects_out_of_range_layer_and_expert() {
        let header = header();
        assert!(header
            .component_offset(2, 0, ExpertComponent::Gate)
            .is_err());
        assert!(header
            .component_offset(79, 0, ExpertComponent::Gate)
            .is_err());
        assert!(header
            .component_offset(3, 256, ExpertComponent::Gate)
            .is_err());
    }
}
