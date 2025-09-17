pub mod camera;
pub mod texture;
use std::mem;

use cgmath::{InnerSpace, Rotation3, Zero};

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use image::GenericImageView;
use wgpu::{util::DeviceExt, wgc::device::queue, wgt::TextureViewDescriptor, Texture, TextureDescriptor, VertexAttribute, VertexBufferLayout};
#[cfg(target_arch = "wasm32")]
use winit::event_loop;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window
};

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw { 
            model: (
                cgmath::Matrix4::from_translation(self.position) *
                cgmath::Matrix4::from(self.rotation)
            ).into()
         }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4]
}

// To Study
impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32;4]>()  as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32;12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4
                }
            ]
        }
    }
}

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use crate::camera::Camera;

pub struct State {
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    clear_color: wgpu::Color,
    window: Arc<Window>,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    diffuse_bind_group: wgpu::BindGroup,

    camera: camera::Camera,
    camera_uniform: camera::CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: camera::CameraController,

    depth_texture: texture::Texture
}

#[repr(C)]
#[derive(Copy, Clone ,Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32;3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>()  as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3   
                },
                VertexAttribute {
                    offset: mem::size_of::<[f32;3 ]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2 
                }
            ]
        }
    }
}

const VERTICES: &[Vertex] = &[ 
/* *
    Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.00759614], }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.43041354], }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.949397], }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.84732914], }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.2652641], }, // E
*/

    // 0
    Vertex {
        position: [0.0, 0.0, 0.0],
        tex_coords: [0.5, 0.5]
        //color: [ 1.0, 0.0, 0.0]
    },
    // 1
    Vertex {
        position: [0.0,0.5,0.0],
        tex_coords: [0.5, 0.75]        
        //color: [ 1.0, 0.0, 0.0]
    } ,
    // 2
    Vertex {
        position: [-0.5,0.0,0.0],
        tex_coords: [0.25, 0.5]        
       // color: [ 1.0, 0.0, 0.0]
    } ,
    // 3
    Vertex {
        position: [-0.5,1.0,0.0],
        tex_coords: [0.4, 0.9]        
        //color: [ 1.0, 0.0, 0.0]
    }  ,
    //4
    Vertex {
        position: [-1.0,0.5,0.0],
        tex_coords: [0.25, 0.75]       
       // color: [ 1.0, 0.0, 0.0]
    }   ,
    //5
    Vertex {
        position: [0.5,1.0,0.0],
        tex_coords: [0.6, 0.9]       
        //color: [ 0.0, 1.0, 0.0]
    }  ,
    //6
    Vertex {
        position: [1.0,0.5,0.0],
        tex_coords: [0.8, 0.75]      
        //color: [ 0.0, 1.0, 0.0]
    } ,
    //7
    Vertex {
        position: [0.5,0.0,0.0],
        tex_coords: [0.75, 0.5]        
       // color: [ 0.0, 1.0, 0.0]
    }  ,
    //8
    Vertex {
        position: [0.5,-0.5,0.0],
        tex_coords: [0.75, 0.25]        
        //color: [ 0.0, 0.0, 1.0]
    },
    // 9
    Vertex {
        position: [0.0,-1.0,0.0],
        tex_coords: [1.0, 0.5]        
        //color: [ 0.0, 0.0, 1.0]
    } ,
    // 10
    Vertex {
        position: [-0.5,-0.5,0.0],
        tex_coords: [0.25, 0.25]        
        //color: [ 0.0, 0.0, 1.0]
    } 

    // 0 - A
    //Vertex { position: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0 ]}, // A
    //// 1 - B
    //Vertex { position: [-1.0, 1.0, 1.0], color: [0.0, 1.0, 0.0] }, // B
    //// 2 - C
    //Vertex { position: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] }, // C
    //// 3 - D
    //Vertex { position: [1.0, 0.0, 0.0], color: [1.0, 0.0, 0.5] }, // D
    //// 4 - E
    //Vertex { position: [0.0, -1.0, 0.0], color: [0.0, 0.5, 1.0] }, // E
    //// 5 - F
    //Vertex { position: [0.44147372, 0.2347359, 0.0], color: [17.5, 18.0, 19.5] }, // F
];

const INDICES: &[u16] = &[
    0, 1, 2,
    1, 3, 2,
    1, 3 ,4,
    1, 4, 2,
    7, 1, 0,
    6,5,1,
    6, 1, 7,
    7, 0, 8,
    8,0, 9,
    8,0,10,
    8, 10, 9,
    0, 2, 10
/* 
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
*/
    ];

// To Study
const NUM_INSTANCES_PER_ROW: u32 = 8;
// To Study
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
    NUM_INSTANCES_PER_ROW as f32 * 0.5, 0.0,NUM_INSTANCES_PER_ROW as f32 * 0.5
);

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {

        let size = window.inner_size();
        let num_vertices = VERTICES.len() as u32;
        let num_indices = INDICES.len() as u32;
        let instance = wgpu::Instance::new(
            &wgpu::InstanceDescriptor {
                #[cfg(not(target_arch="wasm32"))]
                backends: wgpu::Backends::PRIMARY,
                #[cfg(target_arch="wasm32")]
                backends: wgpu::Backends::GL,
                ..Default::default()
            }
        );
        log::info!("backends: {:#?}",wgpu::Backends::all());

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await?;
        log::info!("Adapter Info: {:#?}",adapter.get_info());
        log::info!("Capabilities: {:#?}",surface.get_capabilities(&adapter));

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            }
            else {
                wgpu::Limits::default()
            },
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off
        }).await?;
        log::info!("Device Features: {:#?}",device.features());
        log::info!("Device Limits: {:#?}",device.limits());

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX
            }
        );

        // To study
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX
            }
        );

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2
        };
        
        let diffuse_bytes = include_bytes!("tree.jpg");
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree").unwrap();
        
        
        let texture_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true
                            }
                        },
                        count: None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None
                    }
                ],
                label: Some("texture_bind_group_layout")
            }
        );

        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view)
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler)
                    }
                ],
                label: Some("diffuse_bind_group")
            },
        );

        let camera = Camera {
            eye: (0.0,0.0,3.0).into(),
            target: (0.0,0.0,0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: diffuse_texture.texture.width() as f32 / diffuse_texture.texture.height()  as f32,
            fovy: 50.0,
            znear: 0.1,
            zfar: 100.0,
        };
        
        let mut camera_uniform = camera::CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None
                    }
                ]
            }
        );

        let camera_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }
                ],
                label: Some("camera_bind_group")
            }
        );

        let shader = device.create_shader_module(
            wgpu::ShaderModuleDescriptor{
                label: Some("Shared"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader-instances.wgsl").into()),
            }
        );

        
        // To Study
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let position = cgmath::Vector3 {
                    x: x as f32,
                    y: 0.0,
                    z: z as f32
                } - INSTANCE_DISPLACEMENT;

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                Instance {
                    position, rotation
                }
            })
        }).collect::<Vec<_>>();

        // To Study
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX
            }
        );

        let render_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout
                ],
                push_constant_ranges: &[]
            }
        );

        //To Study
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");
        
        let render_pipeline= device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    //To Study
                    buffers: &[Vertex::desc(), InstanceRaw::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default()
                },
                fragment: Some( wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(wgpu::ColorTargetState { 
                            format: config.format, 
                            blend: Some(wgpu::BlendState::REPLACE), 
                            write_mask: wgpu::ColorWrites::ALL
                        })
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default()
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false
                },
                //To Study
                depth_stencil: Some(wgpu::DepthStencilState { 
                    format: texture::Texture::DEPTH_FORMAT, 
                    depth_write_enabled: true, 
                    depth_compare: wgpu::CompareFunction::Less , 
                    stencil: wgpu::StencilState::default(), 
                    bias: wgpu::DepthBiasState::default() 
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false
                },
                multiview: None,
                cache: None,
            }
        );

        let camera_controller = camera::CameraController::new(0.2);


        Ok(Self {
            surface,
            device,
            queue,
            config,
            clear_color: wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
            is_surface_configured: false,
            window,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            camera,
            camera_bind_group,
            camera_buffer,
            camera_uniform,
            camera_controller,
            instance_buffer,
            instances,
            depth_texture
        })
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        let result = self.camera_controller.process_events(event);
        self.update();
        result
    }

    fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            //To Study
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }
    
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError>{
        self.window.request_redraw();
        
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                    view: &view, 
                    resolve_target: None, 
                    ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            store: wgpu::StoreOp::Store
                        } 
                    }
                )],
                depth_stencil_attachment: Some(
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture.view,
                        depth_ops: Some(wgpu::Operations { 
                            load: wgpu::LoadOp::Clear(1.0), 
                            store: wgpu::StoreOp::Store 
                        }),
                        stencil_ops: None
                    }
                ),
                occlusion_query_set: None,
                timestamp_writes: None
            });

            render_pass.set_pipeline(&self.render_pipeline);
            
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1,&self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            // To Study
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            //render_pass.draw(0..self.num_vertices, 0..1);
            //To Study
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as u32);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }
}

pub struct App {
    #[cfg(target_arch="wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State> ) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<State> for App {
    
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch="wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(proxy.send_event(
                        State::new(window).await.expect("Unable to create canvas")
                    ).is_ok())
                });
            }
        }        
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height
            );
        }
        self.state = Some(event);
    }
    

    fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
        
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return
        };
        state.input(&event);

        match event {
            WindowEvent::CursorMoved { device_id, position } => {
                log::info!("hello cursor moved: x: {}, y: {}",position.x, position.y);
                state.clear_color.r = position.x / state.config.width as f64;
                state.clear_color.b = position.y / state.config.height as f64;
                state.clear_color.g = 0.0;
                state.clear_color.a = 1.0;
            },
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                match state.render() {
                    Ok(_) => {},
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated ) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    },
                    Err(e) => {
                        log::error!("unable to render {}",e);
                    }
                }
            },
            WindowEvent::KeyboardInput { 
                event: KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                match (code, state.is_pressed()) {
                    (KeyCode::Escape, true) => event_loop.exit(),
                    _ => {}
                }
            },
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch="wasm32"))]
    {
        env_logger::init();   
    }

    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch="wasm32")]
        &event_loop
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(target_arch="wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}

