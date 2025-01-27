import Renderer from "./renderer";

let GRID_SIZE = 64;
const WORKGROUP_SIZE = 8;

let targetFPS = 10; // Desired FPS
let frameInterval = 1000 / targetFPS; // Interval in milliseconds
let step = 0;
let isPaused = false;
let currentInitialState: Uint32Array;


const initialStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
for (let i = 0; i < initialStateArray.length; ++i) {
  initialStateArray[i] = Math.random() > 0.6 ? 1 : 0;
}
currentInitialState = initialStateArray;

const canvas = document.querySelector<HTMLCanvasElement>("canvas")!;

const pauseButton = document.querySelector<HTMLButtonElement>("#pause-button");
pauseButton?.addEventListener("click", (_event) => {
  isPaused = !isPaused;
  pauseButton.textContent = isPaused ? "Resume" : "Pause";

  if (!isPaused) {
    requestAnimationFrame(renderLoop);
  }
});

const restartButton = document.querySelector<HTMLButtonElement>("#restart-button");
restartButton?.addEventListener("click", async (_event) => {
  await device.queue.onSubmittedWorkDone();

  step = 0;
  device.queue.writeBuffer(cellStateStorage[0], 0, currentInitialState);
  device.queue.writeBuffer(cellStateStorage[1], 0, currentInitialState);
});

const fpsForm = document.querySelector<HTMLFormElement>("#fps-form");
fpsForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  const fpsInput = document.querySelector<HTMLInputElement>("#fps-input");

  const fpsValue = fpsInput?.value;
  if (fpsValue) {
    targetFPS = parseInt(fpsInput.value);
    frameInterval = 1000 / targetFPS;
  }
});

const gridSizeSelect = document.querySelector<HTMLSelectElement>("#grid-size");
gridSizeSelect?.addEventListener("change", async (event) => {
  const newSize = parseInt((event.target as HTMLSelectElement).value);
  await handleGridSizeChange(newSize);
});

async function handleGridSizeChange(newSize: number) {
  const wasPaused = isPaused;
  isPaused = true;
  pauseButton!.textContent = "Pause";

  await device.queue.onSubmittedWorkDone();

  GRID_SIZE = newSize;

  const newInitialStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
  for (let i = 0; i < newInitialStateArray.length; ++i) {
    newInitialStateArray[i] = Math.random() > 0.6 ? 1 : 0;
  }
  currentInitialState = newInitialStateArray;

  const newGridArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
  device.queue.writeBuffer(gridBuffer, 0, newGridArray);

  cellStateStorage[0].destroy();
  cellStateStorage[1].destroy();

  cellStateStorage[0] = device.createBuffer({
    label: "Cell State",
    size: newInitialStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  cellStateStorage[1] = device.createBuffer({
    label: "Cell State B",
    size: newInitialStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(cellStateStorage[0], 0, newInitialStateArray);
  device.queue.writeBuffer(cellStateStorage[1], 0, newInitialStateArray);

  bindGroups[0] = device.createBindGroup({
    label: "Cell renderer bind group A",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffer } },
      { binding: 1, resource: { buffer: cellStateStorage[0] } },
      { binding: 2, resource: { buffer: cellStateStorage[1] } }
    ],
  });
  bindGroups[1] = device.createBindGroup({
    label: "Cell renderer bind group B",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffer } },
      { binding: 1, resource: { buffer: cellStateStorage[1] } },
      { binding: 2, resource: { buffer: cellStateStorage[0] } },
    ],
  });

  step = 0;
  isPaused = wasPaused;
  pauseButton!.textContent = isPaused ? 'Resume' : 'Pause';

  if (!isPaused) {
    requestAnimationFrame(renderLoop);
  }
}

const { ctx, device } = await Renderer.create(canvas);

const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({
  device: device,
  format: canvasFormat,
});

const gridArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const gridBuffer = device.createBuffer({
  label: "Grid Uniforms",
  size: gridArray.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(gridBuffer, 0, gridArray);

const vertices = new Float32Array([
//   X,    Y,
  -0.8, -0.8,
   0.8, -0.8,
   0.8,  0.8,

  -0.8, -0.8,
   0.8,  0.8,
  -0.8,  0.8,
]);

const vertexBuffer = device.createBuffer({
  label: "Cell vertices",
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);

const vertexBufferLayout = {
  arrayStride: 8,
  attributes: [{
    format: "float32x2",
    offset: 0,
    shaderLocation: 0, // Position, see vertex shader
  }],
} satisfies GPUVertexBufferLayout;

const cellStateStorage = [
  device.createBuffer({
    label: "Cell State",
    size: initialStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }),
  device.createBuffer({
    label: "Cell State B",
     size: initialStateArray.byteLength,
     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }),
];

device.queue.writeBuffer(cellStateStorage[0], 0, initialStateArray);

const cellShaderModule = device.createShaderModule({
  label: "Cell shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellState: array<u32>;

    struct VertexInput {
      @location(0) pos: vec2f,
      @builtin(instance_index) instance: u32,
    };
    
    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) cell: vec2f,
    };

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput
    {
      let i = f32(input.instance);
      let cell = vec2f(i % grid.x, floor(i / grid.x));
      let state = f32(cellState[input.instance]);

      let cellOffset = cell / grid * 2;
      let gridPos = (input.pos*state+1) / grid - 1 + cellOffset;

      var output: VertexOutput;
      output.pos = vec4f(gridPos, 0, 1);
      output.cell = cell;
      return output;
    }

    struct FragInput {
      @location(0) cell: vec2f,
    };

    @fragment
    fn fragmentMain(input: FragInput) -> @location(0) vec4f {
      let c = input.cell / grid;
      return vec4f(c, 1-c.x, 1);
    }
  `
});

const simulationShaderModule = device.createShaderModule({
  label: "Game of Life simulation shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

    fn cellIndex(cell: vec2u) -> u32 {
      return (cell.y % u32(grid.y)) * u32(grid.x) +
            (cell.x % u32(grid.x));
    }

    fn cellActive(x: u32, y: u32) -> u32 {
      return cellStateIn[cellIndex(vec2(x, y))];
    }

    @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
      let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                            cellActive(cell.x+1, cell.y) +
                            cellActive(cell.x+1, cell.y-1) +
                            cellActive(cell.x, cell.y-1) +
                            cellActive(cell.x-1, cell.y-1) +
                            cellActive(cell.x-1, cell.y) +
                            cellActive(cell.x-1, cell.y+1) +
                            cellActive(cell.x, cell.y+1);
      
      let i = cellIndex(cell.xy);
      switch activeNeighbors {
        case 2: {
          cellStateOut[i] = cellStateIn[i];
        }
        case 3: {
          cellStateOut[i] = 1;
        }
        default: {
          cellStateOut[i] = 0;
        }
      }
    }
  `
});

// Create the bind group layout and pipeline layout.
const bindGroupLayout = device.createBindGroupLayout({
  label: "Cell Bind Group Layout",
  entries: [{
    binding: 0,
    visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
    buffer: {} // Grid uniform buffer
  }, {
    binding: 1,
    visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
    buffer: { type: "read-only-storage"} // Cell state input buffer
  }, {
    binding: 2,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: "storage"} // Cell state output buffer
  }]
});

const bindGroups = [
  device.createBindGroup({
    label: "Cell renderer bind group A",
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: gridBuffer }
      },
      {
        binding: 1,
        resource: { buffer: cellStateStorage[0] }
      },
      {
        binding: 2,
        resource: { buffer: cellStateStorage[1] }
      }
    ],
  }),
   device.createBindGroup({
    label: "Cell renderer bind group B",
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: gridBuffer }
      },
      {
        binding: 1,
        resource: { buffer: cellStateStorage[1] }
      },
      {
        binding: 2,
        resource: { buffer: cellStateStorage[0] }
      },
    ],
  })
];

const pipelineLayout = device.createPipelineLayout({
  label: "Cell Pipeline Layout",
  bindGroupLayouts: [ bindGroupLayout ],
});

const cellPipeline = device.createRenderPipeline({
  label: "Cell pipeline",
  layout: pipelineLayout,
  vertex: {
    module: cellShaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout]
  },
  fragment: {
    module: cellShaderModule,
    entryPoint: "fragmentMain",
    targets: [{
      format: canvasFormat,
    }]
  }
});

// Create a compute pipeline that updates the game state.
const simulationPipeline = device.createComputePipeline({
  label: "Simulation pipeline",
  layout: pipelineLayout,
  compute: {
    module: simulationShaderModule,
    entryPoint: "computeMain",
  }
});

const updateGrid = () => {
  const encoder = device.createCommandEncoder();

  const computePass = encoder.beginComputePass();

  computePass.setPipeline(simulationPipeline);
  computePass.setBindGroup(0, bindGroups[step % 2]);

  const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

  computePass.end();

  step++;

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      loadOp: "clear",
      clearValue: { r: 0, g: 0, b: 0.3, a: 1 }, // New line
      storeOp: "store",
    }]
  });

  renderPass.setPipeline(cellPipeline);
  renderPass.setVertexBuffer(0, vertexBuffer);
  renderPass.setBindGroup(0, bindGroups[step % 2]);

  renderPass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);

  renderPass.end();

  // Finish the command buffer and immediately submit it.
  device.queue.submit([encoder.finish()]);
};


let lastFrameTime = 0;

const renderLoop = (timestamp: number) => {
  if (isPaused) return;

  const elapsedTime = timestamp - lastFrameTime;

  if (elapsedTime >= frameInterval) {
    lastFrameTime = timestamp;
    updateGrid(); // Perform the rendering and update logic
  }

  requestAnimationFrame(renderLoop);
};

// Start the render loop
requestAnimationFrame(renderLoop);