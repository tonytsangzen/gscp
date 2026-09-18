//! wgpu 合成渲染管线。
//!
//! 移植自 mixplayer renderer.rs，并补齐欠账：
//! - `overlay_scale` 参数真正生效（原实现硬编码画布高 120%）；
//! - 黑边抠像 key_low/key_high/feather_power/feather_radius 在 shader 中实现；
//! - 底图亮度、overlay 区域压暗强度、饱和度增益参数化（原为 shader 魔数）。
//! 效果参数经 [`gscp_core::settings::EffectParams`] 共享状态热更新。

use anyhow::{anyhow, Context, Result};
use gscp_core::settings::EffectParams;
use std::borrow::Cow;
use std::f32::consts::PI;
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::{
    AddressMode, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, BufferBindingType,
    BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor,
    CompositeAlphaMode, Device, DeviceDescriptor, Extent3d, Features, FilterMode, FragmentState,
    ImageCopyTexture, ImageDataLayout, Instance, InstanceDescriptor, Limits, LoadOp,
    MultisampleState, Operations, Origin3d, PipelineLayoutDescriptor, PowerPreference, PresentMode,
    PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, Surface, SurfaceConfiguration,
    SurfaceError, Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    VertexState,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::slot::SlotSnapshot;

const BUSY_ICON_TEXTURE_SIZE: u32 = 128;
const BUSY_ICON_RENDER_SIZE: u32 = 96;

const SHADER_SOURCE: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct LayerUniform {
    origin: vec2<f32>,
    size: vec2<f32>,
    crop_origin: vec2<f32>,
    crop_size: vec2<f32>,
    alpha: f32,
    enabled: u32,
    brightness: f32,
    key_low: f32,
    key_high: f32,
    feather_power: f32,
    feather_radius: u32,
    rotation_quadrants: u32,
    dim_strength: f32,
    saturation_boost: f32,
    pad2: u32,
    pad3: u32,
};

struct CompositeUniforms {
    base_layer: LayerUniform,
    overlay_layer: LayerUniform,
};

@group(0) @binding(0) var<uniform> uniforms: CompositeUniforms;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var base_tex: texture_2d<f32>;
@group(0) @binding(3) var overlay_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32) -> VertexOut {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );

    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_id], 0.0, 1.0);
    out.uv = uvs[vertex_id];
    return out;
}

fn layer_tex_uv(layer: LayerUniform, uv: vec2<f32>) -> vec2<f32> {
    var tex_uv = (uv - layer.origin) / layer.size;
    tex_uv = layer.crop_origin + tex_uv * layer.crop_size;
    switch (layer.rotation_quadrants % 4u) {
        case 1u: {
            tex_uv = vec2<f32>(1.0 - tex_uv.y, tex_uv.x);
        }
        case 2u: {
            tex_uv = vec2<f32>(1.0 - tex_uv.x, 1.0 - tex_uv.y);
        }
        case 3u: {
            tex_uv = vec2<f32>(tex_uv.y, 1.0 - tex_uv.x);
        }
        default: {}
    }
    return tex_uv;
}

fn sample_layer(
    tex: texture_2d<f32>,
    layer: LayerUniform,
    uv: vec2<f32>,
    fallback_color: vec4<f32>
) -> vec4<f32> {
    if (layer.enabled == 0u) {
        return fallback_color;
    }

    let min_uv = layer.origin;
    let max_uv = layer.origin + layer.size;
    if (uv.x < min_uv.x || uv.x > max_uv.x || uv.y < min_uv.y || uv.y > max_uv.y) {
        return fallback_color;
    }

    let tex_uv = layer_tex_uv(layer, uv);
    return textureSample(tex, tex_sampler, tex_uv);
}

fn overlay_dim_mask(local_uv: vec2<f32>) -> f32 {
    let mask_size = vec2<f32>(1.0, 0.6);
    let mask_half_size = mask_size * 0.5;
    let mask_radius = mask_size.y * 0.08;
    let feather_width = mask_size.x * 0.6;
    let delta = abs(local_uv - vec2<f32>(0.5, 0.5)) - (mask_half_size - vec2<f32>(mask_radius));
    let outside = length(max(delta, vec2<f32>(0.0, 0.0)));
    let inside = min(max(delta.x, delta.y), 0.0);
    let signed_distance = outside + inside - mask_radius;
    return 1.0 - smoothstep(0.0, feather_width, signed_distance);
}

fn boost_saturation(color: vec3<f32>, gain: f32) -> vec3<f32> {
    let luminance = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    let grayscale = vec3<f32>(luminance);
    return clamp(grayscale + (color - grayscale) * gain, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn overlay_texel_size(tex: texture_2d<f32>, layer: LayerUniform) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(tex));
    return 1.0 / max(dims * max(layer.crop_size, vec2<f32>(0.0001, 0.0001)), vec2<f32>(1.0, 1.0));
}

fn sample_overlay_filtered(layer: LayerUniform, tex_uv: vec2<f32>) -> vec4<f32> {
    let texel = overlay_texel_size(overlay_tex, layer);
    let aa_offset = texel * 0.65;
    let center = textureSample(overlay_tex, tex_sampler, tex_uv);
    let aa_color = center * 0.4
        + textureSample(overlay_tex, tex_sampler, tex_uv + vec2<f32>(-aa_offset.x, -aa_offset.y)) * 0.15
        + textureSample(overlay_tex, tex_sampler, tex_uv + vec2<f32>( aa_offset.x, -aa_offset.y)) * 0.15
        + textureSample(overlay_tex, tex_sampler, tex_uv + vec2<f32>(-aa_offset.x,  aa_offset.y)) * 0.15
        + textureSample(overlay_tex, tex_sampler, tex_uv + vec2<f32>( aa_offset.x,  aa_offset.y)) * 0.15;

    let neighbors = (
        textureSample(overlay_tex, tex_sampler, tex_uv + vec2<f32>( texel.x, 0.0)) +
        textureSample(overlay_tex, tex_sampler, tex_uv + vec2<f32>(-texel.x, 0.0)) +
        textureSample(overlay_tex, tex_sampler, tex_uv + vec2<f32>(0.0,  texel.y)) +
        textureSample(overlay_tex, tex_sampler, tex_uv + vec2<f32>(0.0, -texel.y))
    ) * 0.25;
    let edge_alpha = abs(center.a - neighbors.a);
    let edge_luma = abs(dot(center.rgb - neighbors.rgb, vec3<f32>(0.299, 0.587, 0.114)));
    let edge_mask = smoothstep(0.02, 0.12, max(edge_alpha, edge_luma * 1.5));
    let sharpened_rgb = clamp(
        aa_color.rgb + (center.rgb - neighbors.rgb) * 0.8 * edge_mask,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
    return vec4<f32>(sharpened_rgb, aa_color.a);
}

// 黑边抠像：luma 低于 key_low → 全透明，高于 key_high → 不变，
// 中间按 feather_power 曲线羽化；feather_radius 外扩阈值区间。
fn black_key_alpha(color: vec4<f32>, layer: LayerUniform) -> f32 {
    if (layer.key_high <= layer.key_low) {
        return color.a;
    }
    let radius = f32(layer.feather_radius) / 255.0;
    let lo = max(layer.key_low - radius, 0.0);
    let hi = min(layer.key_high + radius, 1.0);
    let luma = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let t = clamp((luma - lo) / max(hi - lo, 0.0001), 0.0, 1.0);
    return color.a * pow(t, layer.feather_power);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var out_color = sample_layer(
        base_tex,
        uniforms.base_layer,
        in.uv,
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
    out_color = vec4<f32>(
        clamp(out_color.rgb * uniforms.base_layer.brightness, vec3<f32>(0.0), vec3<f32>(1.0)),
        out_color.a
    );

    if (uniforms.overlay_layer.enabled == 1u) {
        let min_uv = uniforms.overlay_layer.origin;
        let max_uv = uniforms.overlay_layer.origin + uniforms.overlay_layer.size;
        let local_uv = (in.uv - min_uv) / uniforms.overlay_layer.size;
        let dim_mask = overlay_dim_mask(local_uv);
        var overlay_rgb = vec3<f32>(0.0);

        if (in.uv.x >= min_uv.x && in.uv.x <= max_uv.x && in.uv.y >= min_uv.y && in.uv.y <= max_uv.y) {
            let overlay_tex_uv = layer_tex_uv(uniforms.overlay_layer, in.uv);
            var overlay_color = sample_overlay_filtered(uniforms.overlay_layer, overlay_tex_uv);
            overlay_color = vec4<f32>(overlay_color.rgb, black_key_alpha(overlay_color, uniforms.overlay_layer));
            let overlay_gain = uniforms.overlay_layer.alpha * uniforms.overlay_layer.brightness;
            let overlay_enhanced = clamp(
                boost_saturation(overlay_color.rgb, uniforms.overlay_layer.saturation_boost) * 1.12,
                vec3<f32>(0.0),
                vec3<f32>(1.0)
            );
            overlay_rgb = overlay_enhanced * overlay_color.a * overlay_gain;
        }

        let base_dim = 1.0 - uniforms.overlay_layer.alpha * dim_mask * uniforms.overlay_layer.dim_strength;
        out_color = vec4<f32>(
            clamp(out_color.rgb * base_dim, vec3<f32>(0.0), vec3<f32>(1.0)),
            out_color.a
        );
        out_color = vec4<f32>(clamp(out_color.rgb + overlay_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), out_color.a);
    }

    return vec4<f32>(out_color.rgb, 1.0);
}
"#;

pub trait Renderer {
    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<()>;
    fn render(&mut self, base_snapshot: &SlotSnapshot, overlay_snapshot: &SlotSnapshot) -> Result<()>;
    fn backend_name(&self) -> &'static str;
}

pub fn create_renderer(
    window: Arc<Window>,
    effects: Arc<Mutex<EffectParams>>,
) -> Result<Box<dyn Renderer>> {
    Ok(Box::new(WgpuRenderer::new(window, effects)?))
}

/// canvas 归一化几何。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerUniform {
    origin: [f32; 2],
    size: [f32; 2],
    crop_origin: [f32; 2],
    crop_size: [f32; 2],
    alpha: f32,
    enabled: u32,
    brightness: f32,
    key_low: f32,
    key_high: f32,
    feather_power: f32,
    feather_radius: u32,
    rotation_quadrants: u32,
    dim_strength: f32,
    saturation_boost: f32,
    pad2: u32,
    pad3: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    base_layer: LayerUniform,
    overlay_layer: LayerUniform,
}

impl LayerUniform {
    fn disabled() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    fn from_size(
        render_size: FrameSize,
        draw_size: FrameSize,
        alpha: f32,
        brightness: f32,
        key_low: f32,
        key_high: f32,
        feather_power: f32,
        feather_radius: u32,
        rotation_quadrants: u32,
        dim_strength: f32,
        saturation_boost: f32,
    ) -> Self {
        if render_size.width == 0
            || render_size.height == 0
            || draw_size.width == 0
            || draw_size.height == 0
        {
            return Self::disabled();
        }

        let origin_x = (draw_size.width as f32 - render_size.width as f32) / 2.0;
        let origin_y = (draw_size.height as f32 - render_size.height as f32) / 2.0;
        Self {
            origin: [
                origin_x / draw_size.width as f32,
                origin_y / draw_size.height as f32,
            ],
            size: [
                render_size.width as f32 / draw_size.width as f32,
                render_size.height as f32 / draw_size.height as f32,
            ],
            crop_origin: [0.0, 0.0],
            crop_size: [1.0, 1.0],
            alpha,
            enabled: 1,
            brightness,
            key_low,
            key_high,
            feather_power,
            feather_radius,
            rotation_quadrants,
            dim_strength,
            saturation_boost,
            pad2: 0,
            pad3: 0,
        }
    }

    fn with_crop(mut self, crop_origin: [f32; 2], crop_size: [f32; 2]) -> Self {
        self.crop_origin = crop_origin;
        self.crop_size = crop_size;
        self
    }
}

pub fn normalize_rotation_quadrants(degrees: i32) -> u32 {
    degrees.rem_euclid(360).div_euclid(90) as u32
}

/// 合成管线共享资源（窗口 surface 与离屏目标共用）。
struct CompositeGpu {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
}

fn create_composite_gpu(device: &Device, target_format: TextureFormat) -> Result<CompositeGpu> {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gscp-shader"),
        source: ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("gscp-bind-group-layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("gscp-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("gscp-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(ColorTargetState {
                format: target_format,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview: None,
    });

    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("gscp-sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        ..Default::default()
    });
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gscp-uniform-buffer"),
        size: size_of::<CompositeUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Ok(CompositeGpu {
        pipeline,
        bind_group_layout,
        sampler,
        uniform_buffer,
    })
}

/// 由当前效果参数与帧状态计算 composite uniforms。
fn compute_uniforms(
    canvas_size: FrameSize,
    base_frame: Option<&gscp_core::codec::DecodedFrame>,
    overlay_frame: Option<&gscp_core::codec::DecodedFrame>,
    overlay_fresh: bool,
    fx: &EffectParams,
) -> CompositeUniforms {
    let base_uniform = if let Some(frame) = base_frame {
        let base_rotation = normalize_rotation_quadrants(fx.base_rotate_deg);
        let rotated_size = rotated_frame_size(frame.width, frame.height, base_rotation);
        let (crop_origin, crop_size) = cover_crop_uv(
            rotated_size.width,
            rotated_size.height,
            canvas_size.width,
            canvas_size.height,
        );
        LayerUniform::from_size(
            canvas_size,
            canvas_size,
            1.0,
            fx.base_brightness,
            0.0,
            1.0,
            1.0,
            0,
            base_rotation,
            0.0,
            1.0,
        )
        .with_crop(crop_origin, crop_size)
    } else {
        busy_indicator_uniform(canvas_size)
    };

    let overlay_uniform = if overlay_frame.is_some() && overlay_fresh {
        let frame = overlay_frame.expect("overlay frame");
        let overlay_rotation = normalize_rotation_quadrants(fx.overlay_rotate_deg);
        let rotated_size = rotated_frame_size(frame.width, frame.height, overlay_rotation);
        // overlay 高度 = canvas 高度 × overlay_scale
        let target_height = canvas_size.height.max(1) as f32;
        let scale = target_height / rotated_size.height.max(1) as f32 * fx.overlay_scale;
        let render_size = FrameSize {
            width: ((rotated_size.width as f32) * scale).round().max(1.0) as u32,
            height: ((rotated_size.height as f32) * scale).round().max(1.0) as u32,
        };
        LayerUniform::from_size(
            render_size,
            canvas_size,
            fx.overlay_alpha.clamp(0.0, 1.0),
            fx.overlay_brightness,
            fx.overlay_black_key_low,
            fx.overlay_black_key_high,
            fx.overlay_feather_power,
            fx.overlay_feather_radius,
            overlay_rotation,
            fx.dim_strength,
            fx.saturation_boost,
        )
    } else {
        LayerUniform::disabled()
    };

    CompositeUniforms {
        base_layer: base_uniform,
        overlay_layer: overlay_uniform,
    }
}

pub fn rotated_frame_size(width: u32, height: u32, rotation_quadrants: u32) -> FrameSize {
    if rotation_quadrants % 2 == 1 {
        FrameSize {
            width: height,
            height: width,
        }
    } else {
        FrameSize { width, height }
    }
}

pub fn cover_crop_uv(
    src_width: u32,
    src_height: u32,
    target_width: u32,
    target_height: u32,
) -> ([f32; 2], [f32; 2]) {
    let src_aspect = src_width as f32 / src_height as f32;
    let target_aspect = target_width as f32 / target_height as f32;

    if src_aspect > target_aspect {
        let visible_width = src_height as f32 * target_aspect;
        let crop_x = ((src_width as f32 - visible_width) / 2.0) / src_width as f32;
        let crop_w = visible_width / src_width as f32;
        ([crop_x, 0.0], [crop_w, 1.0])
    } else {
        let visible_height = src_width as f32 / target_aspect;
        let crop_y = ((src_height as f32 - visible_height) / 2.0) / src_height as f32;
        let crop_h = visible_height / src_height as f32;
        ([0.0, crop_y], [1.0, crop_h])
    }
}

struct GpuLayerState {
    texture: Option<Texture>,
    view: Option<TextureView>,
    size: FrameSize,
    seq: u64,
}

impl GpuLayerState {
    fn new() -> Self {
        Self {
            texture: None,
            view: None,
            size: FrameSize::default(),
            seq: 0,
        }
    }

    fn sync_frame(
        &mut self,
        device: &Device,
        queue: &Queue,
        frame: &gscp_core::codec::DecodedFrame,
    ) -> Result<()> {
        let frame_size = FrameSize {
            width: frame.width,
            height: frame.height,
        };
        if self.texture.is_none() || self.size != frame_size {
            let (texture, view) = create_video_texture(device, frame.width, frame.height)?;
            self.texture = Some(texture);
            self.view = Some(view);
            self.size = frame_size;
            self.seq = 0;
        }

        if self.seq != frame.seq {
            let texture = self
                .texture
                .as_ref()
                .ok_or_else(|| anyhow!("GPU 纹理尚未创建"))?;
            upload_rgba_texture(queue, texture, frame.width, frame.height, &frame.pixels)?;
            self.seq = frame.seq;
        }

        Ok(())
    }

    fn view(&self) -> Option<&TextureView> {
        self.view.as_ref()
    }
}

struct WgpuRenderer {
    _window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    surface_config: SurfaceConfiguration,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    placeholder_texture: Texture,
    placeholder_view: TextureView,
    base_gpu: GpuLayerState,
    overlay_gpu: GpuLayerState,
    effects: Arc<Mutex<EffectParams>>,
    canvas_size: FrameSize,
    started_at: Instant,
}

impl WgpuRenderer {
    fn new(window: Arc<Window>, effects: Arc<Mutex<EffectParams>>) -> Result<Self> {
        debug_assert_eq!(size_of::<LayerUniform>(), 80);
        debug_assert_eq!(size_of::<CompositeUniforms>(), 160);

        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .context("创建 wgpu surface 失败")?;
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("未找到可用的 GPU 适配器"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("gscp-device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
            },
            None,
        ))
        .context("创建 wgpu device 失败")?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|mode| *mode == CompositeAlphaMode::Auto)
            .unwrap_or(caps.alpha_modes[0]);
        let surface_config = build_surface_config(window.inner_size(), surface_format, alpha_mode);
        surface.configure(&device, &surface_config);

        let gpu = create_composite_gpu(&device, surface_format)?;
        let (placeholder_texture, placeholder_view) = create_placeholder_texture(&device, &queue)?;

        let size = window.inner_size();
        Ok(Self {
            _window: window,
            surface,
            device,
            queue,
            surface_config,
            bind_group_layout: gpu.bind_group_layout,
            pipeline: gpu.pipeline,
            sampler: gpu.sampler,
            uniform_buffer: gpu.uniform_buffer,
            placeholder_texture,
            placeholder_view,
            base_gpu: GpuLayerState::new(),
            overlay_gpu: GpuLayerState::new(),
            effects,
            canvas_size: FrameSize {
                width: size.width.max(1),
                height: size.height.max(1),
            },
            started_at: Instant::now(),
        })
    }

    fn reconfigure_surface(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.canvas_size = FrameSize {
            width: size.width.max(1),
            height: size.height.max(1),
        };
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn render_pass(
        &mut self,
        base_snapshot: &SlotSnapshot,
        overlay_snapshot: &SlotSnapshot,
    ) -> Result<()> {
        let fx = *self.effects.lock().expect("effects poisoned");

        if let Some(frame) = base_snapshot.frame.as_ref() {
            self.base_gpu.sync_frame(&self.device, &self.queue, frame)?;
        } else {
            self.update_busy_indicator()?;
        }
        if let Some(frame) = overlay_snapshot.frame.as_ref() {
            self.overlay_gpu.sync_frame(&self.device, &self.queue, frame)?;
        }
        let overlay_fresh = overlay_snapshot.frame.is_some()
            && overlay_snapshot
                .last_frame_at
                .map(|at| at.elapsed() <= crate::slot::OVERLAY_HIDE_AFTER_IDLE)
                .unwrap_or(false);
        let uniforms = compute_uniforms(
            self.canvas_size,
            base_snapshot.frame.as_ref(),
            overlay_snapshot.frame.as_ref(),
            overlay_fresh,
            &fx,
        );
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let surface_frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            Err(SurfaceError::Timeout) => return Ok(()),
            Err(SurfaceError::OutOfMemory) => {
                return Err(anyhow!("获取渲染表面失败: GPU 内存不足"));
            }
        };
        let surface_view = surface_frame
            .texture
            .create_view(&TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("gscp-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        self.base_gpu.view().unwrap_or(&self.placeholder_view),
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(
                        self.overlay_gpu.view().unwrap_or(&self.placeholder_view),
                    ),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("gscp-command-encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("gscp-render-pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        surface_frame.present();
        Ok(())
    }

    fn update_busy_indicator(&self) -> Result<()> {
        let elapsed_secs = self.started_at.elapsed().as_secs_f32();
        let bytes = render_busy_icon_texture(BUSY_ICON_TEXTURE_SIZE, elapsed_secs);
        upload_rgba_texture(
            &self.queue,
            &self.placeholder_texture,
            BUSY_ICON_TEXTURE_SIZE,
            BUSY_ICON_TEXTURE_SIZE,
            &bytes,
        )
    }
}

impl Renderer for WgpuRenderer {
    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        self.reconfigure_surface(size);
        Ok(())
    }

    fn render(
        &mut self,
        base_snapshot: &SlotSnapshot,
        overlay_snapshot: &SlotSnapshot,
    ) -> Result<()> {
        self.render_pass(base_snapshot, overlay_snapshot)
    }

    fn backend_name(&self) -> &'static str {
        let _ = &self.placeholder_texture;
        "wgpu"
    }
}

fn build_surface_config(
    size: PhysicalSize<u32>,
    format: TextureFormat,
    alpha_mode: CompositeAlphaMode,
) -> SurfaceConfiguration {
    SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: PresentMode::Fifo,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    }
}

fn busy_indicator_uniform(canvas_size: FrameSize) -> LayerUniform {
    let render_size = FrameSize {
        width: BUSY_ICON_RENDER_SIZE.min(canvas_size.width.max(1)),
        height: BUSY_ICON_RENDER_SIZE.min(canvas_size.height.max(1)),
    };
    LayerUniform::from_size(render_size, canvas_size, 1.0, 1.0, 0.0, 1.0, 1.0, 0, 0, 0.0, 1.0)
}

fn render_busy_icon_texture(size: u32, elapsed_secs: f32) -> Vec<u8> {
    let mut pixels = vec![0_u8; (size as usize) * (size as usize) * 4];
    let center = (size as f32 - 1.0) * 0.5;
    let orbit_radius = size as f32 * 0.28;
    let dot_radius = size as f32 * 0.055;
    let highlight = ((elapsed_secs * 10.0) as usize) % 12;
    let tau = PI * 2.0;

    for y in 0..size {
        for x in 0..size {
            let fx = x as f32;
            let fy = y as f32;
            let mut brightness = 0.0_f32;

            for idx in 0..12 {
                let angle = (idx as f32 / 12.0) * tau - elapsed_secs * 3.0;
                let dot_x = center + angle.cos() * orbit_radius;
                let dot_y = center + angle.sin() * orbit_radius;
                let dx = fx - dot_x;
                let dy = fy - dot_y;
                let distance = (dx * dx + dy * dy).sqrt();
                if distance <= dot_radius {
                    let trail = (12 + idx - highlight) % 12;
                    let level = 0.15 + 0.85 * ((12 - trail) as f32 / 12.0);
                    let edge = 1.0 - (distance / dot_radius);
                    brightness = brightness.max(level * edge);
                }
            }

            let pixel = ((y * size + x) * 4) as usize;
            let value = (brightness.clamp(0.0, 1.0) * 255.0) as u8;
            pixels[pixel] = value;
            pixels[pixel + 1] = value;
            pixels[pixel + 2] = value;
            pixels[pixel + 3] = 255;
        }
    }

    pixels
}

fn create_video_texture(device: &Device, width: u32, height: u32) -> Result<(Texture, TextureView)> {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("gscp-video-texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    Ok((texture, view))
}

fn create_placeholder_texture(device: &Device, queue: &Queue) -> Result<(Texture, TextureView)> {
    let (texture, view) =
        create_video_texture(device, BUSY_ICON_TEXTURE_SIZE, BUSY_ICON_TEXTURE_SIZE)?;
    let bytes = render_busy_icon_texture(BUSY_ICON_TEXTURE_SIZE, 0.0);
    upload_rgba_texture(
        queue,
        &texture,
        BUSY_ICON_TEXTURE_SIZE,
        BUSY_ICON_TEXTURE_SIZE,
        &bytes,
    )?;
    Ok((texture, view))
}

fn upload_rgba_texture(
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
    bytes: &[u8],
) -> Result<()> {
    let expected = gscp_core::codec::rgba_bytes(width, height)?;
    if bytes.len() != expected {
        return Err(anyhow!(
            "纹理上传字节数不匹配，期望 {expected}，实际 {}",
            bytes.len()
        ));
    }

    queue.write_texture(
        ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytes,
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

/// 离屏合成一帧并读回 RGBA（测试/诊断用，不依赖窗口）。
pub fn render_composite_offscreen(
    width: u32,
    height: u32,
    base: Option<&gscp_core::codec::DecodedFrame>,
    overlay: Option<&gscp_core::codec::DecodedFrame>,
    fx: &EffectParams,
) -> Result<Vec<u8>> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow!("未找到可用的 GPU 适配器"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &DeviceDescriptor {
            label: Some("gscp-offscreen"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
        },
        None,
    ))?;

    let gpu = create_composite_gpu(&device, TextureFormat::Rgba8Unorm)?;
    let canvas = FrameSize { width, height };

    let mut base_state = GpuLayerState::new();
    let mut overlay_state = GpuLayerState::new();
    if let Some(frame) = base {
        base_state.sync_frame(&device, &queue, frame)?;
    }
    if let Some(frame) = overlay {
        overlay_state.sync_frame(&device, &queue, frame)?;
    }

    let uniforms = compute_uniforms(canvas, base, overlay, overlay.is_some(), fx);
    queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

    let target = device.create_texture(&TextureDescriptor {
        label: Some("gscp-offscreen-target"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&TextureViewDescriptor::default());

    // 缺省层的 1x1 占位纹理（黑色，enabled=0 时 shader 不会采样）
    let placeholder = device.create_texture(&TextureDescriptor {
        label: Some("gscp-offscreen-placeholder"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &placeholder,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &[0, 0, 0, 255],
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let placeholder_view = placeholder.create_view(&TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("gscp-offscreen-bind-group"),
        layout: &gpu.bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: gpu.uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&gpu.sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(
                    base_state.view().unwrap_or(&placeholder_view),
                ),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(
                    overlay_state.view().unwrap_or(&placeholder_view),
                ),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("gscp-offscreen-encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("gscp-offscreen-pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    // 读回（256 字节对齐行距）
    let bytes_per_row = ((width * 4).div_ceil(256)) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gscp-offscreen-readback"),
        size: (bytes_per_row * height) as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::ImageCopyBuffer {
            buffer: &readback,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = std::sync::mpsc::channel();
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv_timeout(Duration::from_secs(5))
        .map_err(|e| anyhow!("等待 GPU 读回超时: {e}"))?
        .map_err(|e| anyhow!("GPU 读回失败: {e}"))?;
    let data = slice.get_mapped_range().to_vec();
    readback.unmap();

    // 去行距
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * bytes_per_row) as usize;
        out.extend_from_slice(&data[start..start + (width * 4) as usize]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_quadrants() {
        assert_eq!(normalize_rotation_quadrants(90), 1);
        assert_eq!(normalize_rotation_quadrants(-90), 3);
        assert_eq!(normalize_rotation_quadrants(450), 1);
        assert_eq!(normalize_rotation_quadrants(0), 0);
    }

    #[test]
    fn rotated_size_swaps_on_odd_quadrants() {
        assert_eq!(
            rotated_frame_size(1920, 1080, 1),
            FrameSize { width: 1080, height: 1920 }
        );
        assert_eq!(
            rotated_frame_size(1920, 1080, 2),
            FrameSize { width: 1920, height: 1080 }
        );
    }

    #[test]
    fn cover_crop_wide_source() {
        // 源比目标更宽 → 左右居中裁剪：(1920-1080)/2/1920 = 0.21875
        let (origin, size) = cover_crop_uv(1920, 1080, 1080, 1080);
        assert!((origin[0] - 0.21875).abs() < 1e-6);
        assert_eq!(size, [0.5625, 1.0]);
        // 源更窄 → 上下居中裁剪
        let (origin, size) = cover_crop_uv(1080, 1920, 1080, 1080);
        assert_eq!(origin, [0.0, 0.21875]);
        assert_eq!(size, [1.0, 0.5625]);
    }

    #[test]
    fn uniform_layout_80_bytes() {
        assert_eq!(size_of::<LayerUniform>(), 80);
        assert_eq!(size_of::<CompositeUniforms>(), 160);
    }

    #[test]
    fn layer_uniform_centers_and_normalizes() {
        let layer = LayerUniform::from_size(
            FrameSize { width: 640, height: 480 },
            FrameSize { width: 1280, height: 720 },
            0.5,
            1.2,
            0.1,
            0.9,
            1.0,
            8,
            1,
            0.6,
            1.45,
        );
        assert!((layer.origin[0] - 0.25).abs() < 1e-6);
        assert!((layer.size[1] - 480.0 / 720.0).abs() < 1e-6);
        assert_eq!(layer.enabled, 1);
        assert_eq!(layer.rotation_quadrants, 1);
    }

    // ── 离屏合成数值验证（wgpu 后端可用时运行）──

    fn frame(width: u32, height: u32, pixels: Vec<u8>) -> gscp_core::codec::DecodedFrame {
        gscp_core::codec::rgba_frame(width, height, 1, pixels).unwrap()
    }

    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> gscp_core::codec::DecodedFrame {
        frame(width, height, rgba.iter().copied().cycle().take((width * height * 4) as usize).collect())
    }

    fn pixel_at(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let off = ((y * width + x) * 4) as usize;
        [rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3]]
    }

    fn gpu_available() -> bool {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });
        pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_some()
    }

    #[test]
    fn offscreen_base_only_renders_input_colors() {
        if !gpu_available() {
            eprintln!("无 GPU 适配器，跳过");
            return;
        }
        // 左红右蓝底图（64x48 → canvas 64x48，无旋转）
        let mut pixels = vec![0_u8; 64 * 48 * 4];
        for y in 0..48 {
            for x in 0..64 {
                let off = ((y * 64 + x) * 4) as usize;
                if x < 32 {
                    pixels[off] = 220; // R
                } else {
                    pixels[off + 2] = 220; // B
                }
                pixels[off + 3] = 255;
            }
        }
        let base = frame(64, 48, pixels);
        let fx = EffectParams {
            base_rotate_deg: 0,
            base_brightness: 1.0,
            ..Default::default()
        };
        let out = render_composite_offscreen(64, 48, Some(&base), None, &fx).unwrap();
        // 左侧偏红，右侧偏蓝
        let left = pixel_at(&out, 64, 8, 24);
        let right = pixel_at(&out, 64, 56, 24);
        assert!(left[0] > 200 && left[2] < 60, "左半应偏红: {left:?}");
        assert!(right[2] > 200 && right[0] < 60, "右半应偏蓝: {right:?}");
    }

    #[test]
    fn offscreen_black_key_makes_dark_overlay_transparent() {
        if !gpu_available() {
            eprintln!("无 GPU 适配器，跳过");
            return;
        }
        // 底图：绿色
        let base = solid(64, 48, [0, 200, 0, 255]);
        // overlay：左半黑右半白，居中覆盖（scale 使高度 = canvas 高，宽按比例 64）
        let mut ov = vec![0_u8; 32 * 24 * 4];
        for y in 0..24 {
            for x in 0..32 {
                let off = ((y * 32 + x) * 4) as usize;
                if x < 16 {
                    // 黑
                } else {
                    ov[off] = 255;
                    ov[off + 1] = 255;
                    ov[off + 2] = 255;
                }
                ov[off + 3] = 255;
            }
        }
        let overlay = frame(32, 24, ov);

        let fx = EffectParams {
            base_rotate_deg: 0,
            base_brightness: 1.0,
            overlay_scale: 1.0,
            overlay_alpha: 1.0,
            overlay_brightness: 1.0,
            dim_strength: 0.0,
            saturation_boost: 1.0,
            overlay_black_key_low: 0.1,
            overlay_black_key_high: 0.2,
            ..Default::default()
        };
        let out = render_composite_offscreen(64, 48, Some(&base), Some(&overlay), &fx).unwrap();

        // 左侧 overlay 是黑色 → 抠掉 → 底图绿透出
        let left = pixel_at(&out, 64, 12, 24);
        assert!(left[1] > 150 && left[0] < 80 && left[2] < 80, "黑边应被抠掉露出绿底: {left:?}");
        // 右侧 overlay 白色 → 保留（饱和度 1.0、亮度 1.0 → 接近白）
        let right = pixel_at(&out, 64, 48, 24);
        assert!(right[0] > 200 && right[1] > 200 && right[2] > 200, "白色应保留: {right:?}");
    }
}
