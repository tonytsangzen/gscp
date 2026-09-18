//! 解码后端选择。
//!
//! - macOS：VideoToolbox（系统硬解，无第三方依赖），失败自动回退软件解码；
//! - Windows：Media Foundation H.264 MFT（DXVA 硬解，系统内置），失败自动回退
//!   软件解码；
//! - Linux：OpenH264 源码内置编译的软件解码（H264）。

pub mod software;
#[cfg(target_os = "macos")]
pub mod videotoolbox;
#[cfg(target_os = "windows")]
pub mod mediafoundation;

use crate::decode::software::SoftwareDecoder;
use anyhow::Result;
use gscp_core::codec::VideoDecoder;
use gscp_core::settings::DecoderBackend;

/// 按后端策略创建解码器。`Auto`：macOS 优先 VideoToolbox，Windows 优先
/// Media Foundation，均失败时回退 OpenH264 软件解码。
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
        DecoderBackend::MediaFoundation => {
            #[cfg(target_os = "windows")]
            {
                crate::decode::mediafoundation::MediaFoundationDecoder::new()
                    .map(|d| Box::new(d) as Box<dyn VideoDecoder>)
            }
            #[cfg(not(target_os = "windows"))]
            {
                anyhow::bail!("Media Foundation 后端仅在 Windows 可用")
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
            #[cfg(target_os = "windows")]
            {
                match crate::decode::mediafoundation::MediaFoundationDecoder::new() {
                    Ok(d) => Ok(Box::new(d) as Box<dyn VideoDecoder>),
                    Err(err) => {
                        gscp_core::logbus::emit(&format!(
                            "[decoder] Media Foundation 不可用，回退软件解码: {err:#}"
                        ));
                        Ok(Box::new(SoftwareDecoder::new()))
                    }
                }
            }
            #[cfg(not(any(target_os = "macos", target_os = "windows")))]
            {
                Ok(Box::new(SoftwareDecoder::new()))
            }
        }
    }
}
