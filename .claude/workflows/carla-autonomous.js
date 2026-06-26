export const meta = {
  name: 'carla-autonomous',
  description: 'Autonomous CARLA self-driving training with live monitoring and fan-out subagents',
  phases: [
    { title: 'Scan', detail: 'Scan repo structure and current state' },
    { title: 'Diagnose', detail: 'Diagnose GPU/utilization issues' },
    { title: 'Monitor', detail: 'Live training monitoring' },
    { title: 'Fix', detail: 'Apply fixes' }
  ],
}

const REPO = args.repo_path

phase('Scan')
const scan = await agent(`Scan CARLA repo at ${REPO}. List all directories, find STATE.md, HANDOFF.md, CHANGELOG.md, core/ modules, and all trainer scripts (1_TRAIN_*.py). Return JSON with file structure and current state summary.`, {schema: {
  type: 'object',
  properties: {
    directories: {type: 'array', items: {type: 'string'}},
    state_md_exists: {type: 'boolean'},
    handoff_md_exists: {type: 'boolean'},
    core_modules: {type: 'array', items: {type: 'string'}},
    trainer_scripts: {type: 'array', items: {type: 'string'}},
    summary: {type: 'string'}
  }
}})

phase('Diagnose')
const gpu = await agent(`Check GPU and CARLA status. Run: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv. Check if CARLA running: Test-NetConnection localhost -Port ${args.carla_port}. Check Python training processes. Return JSON with GPU utilization %, VRAM used/total, CARLA running status.`, {schema: {
  type: 'object',
  properties: {
    gpu_name: {type: 'string'},
    gpu_util: {type: 'number'},
    vram_used_mb: {type: 'number'},
    vram_total_mb: {type: 'number'},
    carla_running: {type: 'boolean'},
    training_processes: {type: 'array', items: {type: 'string'}},
    issue: {type: 'string'}
  }
}})

const checkpoints = await agent(`List all checkpoints in ${REPO}/master_suite/checkpoints/. Find latest v15 and v14 checkpoints with timestamps. Return JSON with latest checkpoints by version.`, {schema: {
  type: 'object',
  properties: {
    latest_v15: {type: 'object'},
    latest_v14: {type: 'object'},
    total_checkpoints: {type: 'number'}
  }
}})

phase('Monitor')
// Check if training is currently running and get metrics
const trainingOutput = await agent(`Read the latest training output. Check these files for active training:
1. ${REPO}/master_suite/training_live.log (last 50 lines)
2. Any .output files in AppData temp
3. ${REPO}/master_suite/training_carla_v15_log.txt (last 30 lines)

Return the latest training metrics: timesteps, reward, distance, iterations.`, {schema: {
  type: 'object',
  properties: {
    training_active: {type: 'boolean'},
    timesteps: {type: 'number'},
    reward: {type: 'number'},
    distance: {type: 'number'},
    iterations: {type: 'number'},
    latest_log: {type: 'string'}
  }
}})

phase('Fix')
// Based on diagnosis, determine what fixes are needed
const issues = []

// GPU utilization issue
if (gpu.gpu_util < 50) {
  issues.push('LOW_GPU_UTIL')
}

// Training not moving
if (trainingOutput.training_active && trainingOutput.distance < 50) {
  issues.push('CAR_NOT_MOVING')
}

// Return consolidated diagnosis
return {
  scan,
  gpu,
  checkpoints,
  training: trainingOutput,
  issues,
  next_action: issues.length > 0 ? issues[0] : 'MONITOR_ONLY'
}
