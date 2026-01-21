// Backend registry for runtime device discovery
// Provides two-level checking: compile-time (cargo features) + runtime (device availability)

/// Registry для обнаружения доступных backend'ов во время выполнения
#[derive(Debug, Clone)]
pub struct BackendRegistry {
    pub cpu: bool,
    pub metal: bool,
    pub cuda: bool,
}

impl BackendRegistry {
    /// Обнаружить доступные backend'ы при старте интерпретатора
    pub fn detect() -> Self {
        Self {
            cpu: true,  // CPU всегда доступен
            metal: Self::detect_metal(),
            cuda: Self::detect_cuda(),
        }
    }

    /// Обнаружить Metal backend (двухуровневая проверка)
    fn detect_metal() -> bool {
        #[cfg(feature = "metal")]
        {
            // Compile-time: код есть (metal feature включает gpu)
            // Runtime: проверяем доступность устройства
            candle_core::Device::new_metal(0).is_ok()
        }

        #[cfg(not(feature = "metal"))]
        {
            // Compile-time: кода нет
            false
        }
    }

    /// Обнаружить CUDA backend (двухуровневая проверка)
    fn detect_cuda() -> bool {
        #[cfg(feature = "cuda")]
        {
            // Compile-time: код есть (cuda feature включает gpu)
            // Runtime: проверяем доступность устройства
            candle_core::Device::new_cuda(0).is_ok()
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Compile-time: кода нет
            false
        }
    }

    /// Получить список доступных устройств для пользователя
    pub fn available_devices(&self) -> Vec<&'static str> {
        let mut devices = vec!["cpu"];
        if self.metal {
            devices.push("metal");
        }
        if self.cuda {
            devices.push("cuda");
        }
        devices
    }

    /// Автоматически выбрать лучшее доступное устройство
    pub fn auto_select(&self) -> &'static str {
        if self.metal {
            "metal"
        } else if self.cuda {
            "cuda"
        } else {
            "cpu"
        }
    }
}
