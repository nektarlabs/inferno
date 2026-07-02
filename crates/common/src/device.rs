use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    Metal,
    Reference,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal => f.write_str("metal"),
            Self::Reference => f.write_str("reference"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    Metal,
    Cuda,
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Metal => f.write_str("metal"),
            Self::Cuda => f.write_str("cuda"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceReport {
    pub backend: BackendKind,
    pub device: DeviceKind,
    pub custom_kernels: bool,
}

impl DeviceReport {
    pub fn metal() -> Self {
        Self {
            backend: BackendKind::Metal,
            device: DeviceKind::Metal,
            custom_kernels: true,
        }
    }

    pub fn reference(device: DeviceKind) -> Self {
        Self {
            backend: BackendKind::Reference,
            device,
            custom_kernels: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_report_displays_lowercase_fields() {
        let report = DeviceReport::metal();

        assert_eq!(report.backend.to_string(), "metal");
        assert_eq!(report.device.to_string(), "metal");
        assert!(report.custom_kernels);
    }
}
