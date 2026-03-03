// WebSocket server startup

use crate::infra::cli::WebSocketConfig;

/// Start WebSocket server with given configuration
pub fn start_websocket_server(config: WebSocketConfig) -> Result<(), String> {
    let address = format!("{}:{}", config.host, config.port);
    
    println!("🚀 Запуск WebSocket сервера DataCode...");
    println!("📡 Адрес: ws://{}", address);
    if config.use_ve {
        println!("📁 Режим виртуальной среды: включен (--use-ve)");
    }
    if config.build_model {
        println!("🗄️  Режим экспорта SQLite: включен (--build_model)");
    }
    println!("💡 Используйте --host и --port для изменения адреса");
    println!("💡 Или переменную окружения DATACODE_WS_ADDRESS");
    println!();
    
    // Create tokio runtime for async execution
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;
    
    rt.block_on(crate::websocket::start_server(&address, config.use_ve, config.build_model))
        .map_err(|e| format!("Ошибка запуска WebSocket сервера: {}", e))
}
