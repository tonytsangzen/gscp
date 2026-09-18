//! 解码后端选择。
//!
//! - macOS：VideoToolbox（系统硬解，无第三方依赖），失败自动回退软件解码；
//! - Linux / Windows：OpenH264 源码内置编译的软件解码（H264）。

pub mod software;
#[cfg(target_os = "macos")]
pub mod videotoolbox;

use crate::decode::software::SoftwareDecoder;
use anyhow::Result;
use gscp_core::codec::VideoDecoder;
use gscp_core::settings::DecoderBackend;

/// 按后端策略创建解码器。`Auto` 在 macOS 上优先 VideoToolbox。
pub fn create_decoder(backend: DecoderBackend) -> Result<Box<dyn VideoDecoder>> {
    match backend {
        DecoderBackend::Software => Ok(Box::new(SoftwareDecoder::new())),
        DecoderBackend::VideoToolbox => {
            #[cfg(target_os = "macos")]
            {
                crate::decode::videotoolbox::VideoToolboxDecoder::new()
                    .map(|d| Box::new(d) as Box<dyn VideoDecoder>)
            }
            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("VideoToolbox 后端仅在 macOS 可用")
            }
        }
        DecoderBackend::Auto => {
            #[cfg(target_os = "macos")]
            {
                match crate::decode::videotoolbox::VideoToolboxDecoder::new() {
                    Ok(d) => Ok(Box::new(d) as Box<dyn VideoDecoder>),
                    Err(err) => {
                        gscp_core::logbus::emit(&format!(
                            "[decoder] VideoToolbox 不可用，回退软件解码: {err:#}"
                        ));
                        Ok(Box::new(SoftwareDecoder::new()))
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                Ok(Box::new(SoftwareDecoder::new()))
            }
        }
    }
}
