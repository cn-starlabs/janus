# Janus Blender Workflow — Step-by-Step Guide

> **Purpose**: This guide walks you through the complete Janus CFD workflow inside Blender,
> from scene creation to live simulation to output export.  
> **Prerequisite**: The addon is installed and `libjanus_ffi.so` is built.
> See [BLENDER_ADDON_GUIDE.md](BLENDER_ADDON_GUIDE.md) for installation instructions.

---

## Quick-Start Checklist

| # | Step | Panel | Time |
|---|------|-------|------|
| 1 | Create domain grid mesh | N/A (mesh tools) | ~1 min |
| 2 | Place boundary marker objects | N/A (mesh tools) | ~2 min |
| 3 | Assign boundary roles | **Janus CFD → Setup** | ~30 s |
| 4 | Configure solver parameters | **Janus CFD → Simulate** | ~1 min |
| 5 | Setup visualization materials | **Janus CFD → Visualize** | ~10 s |
| 6 | Start simulation | **Janus CFD → Simulate** | live |
| 7 | Watch live field updates | Viewport | live |
| 8 | Stop and export | **Janus CFD → Simulate** | ~5 s |

---

## Stage 1: Create the Domain Grid

The domain grid is the mesh that Janus uses to display field data (density,
temperature, Kn number, etc.) in the Blender viewport.

### 1.1 Open a New Blender Scene

1. Open Blender and start with the default scene or open your `.blend` file.
2. Switch the viewport to **Solid** or **Material Preview** shading.

### 1.2 Create the Grid Mesh

1. Delete the default cube: select it, press `X` → **Delete**.
2. Add a grid: **Add → Mesh → Grid**.
3. In the **Add Grid** panel (bottom-left), set:
   - **X Subdivisions**: `64`
   - **Y Subdivisions**: `64`
   - **Size**: `1.0` (represents the physical domain side, e.g., 1 mm)
4. Rename the object to `JanusField`:
   - In the **Outliner** (top-right), double-click the object name → type `JanusField`.

> **Why 64×64?** This gives 4 096 cells — enough for good resolution while
> staying fast on a single core. Scale up once you've confirmed the workflow.

### 1.3 Verify the Grid

- Open the **Properties** panel → **Object Data** tab.
- Confirm the mesh has `64 × 64 = 4 096` faces.

```python
# Validation snippet (run in Blender Scripting tab)
import bpy
obj = bpy.data.objects["JanusField"]
assert len(obj.data.polygons) == 64 * 64, "Wrong face count"
print("✓ Grid verified")
```

---

## Stage 2: Place Boundary Marker Objects

Boundary markers are simple mesh objects (cubes work well) placed just
**outside** the domain at each edge: west, east, south, north.

### 2.1 Add West Boundary

1. **Add → Mesh → Cube**, scale it down: `S → 0.05 → Enter`.
2. Move it to the left of the domain: `G → X → -0.6 → Enter`.
3. Rename it `Boundary_West` in the Outliner.

### 2.2 Add East, South, North Boundaries

Repeat the steps above with these positions (for a 1 m × 1 m domain at origin):

| Role  | Name              | Position (X, Y, Z) |
|-------|-------------------|---------------------|
| West  | `Boundary_West`   | (−0.6, 0.5, 0)      |
| East  | `Boundary_East`   | (1.6, 0.5, 0)       |
| South | `Boundary_South`  | (0.5, −0.6, 0)      |
| North | `Boundary_North`  | (0.5, 1.6, 0)       |

> **Tip**: Position doesn't need to be exact — the addon infers the role from
> which domain edge each object is closest to. See Stage 3.

### 2.3 Verify Marker Placement

```python
import bpy
markers = ["Boundary_West", "Boundary_East", "Boundary_South", "Boundary_North"]
for name in markers:
    assert name in bpy.data.objects, f"Missing: {name}"
print("✓ All 4 boundary markers present")
```

---

## Stage 3: Assign Boundary Roles (Janus CFD Panel)

### 3.1 Open the Janus Panel

In the **3D Viewport**, press `N` to open the sidebar, then click the
**Janus CFD** tab.

### 3.2 Set Library Path (first-time only)

1. Go to **Edit → Preferences → Add-ons → Janus CFD**.
2. In the **Janus FFI Library** field, set the path to:
   ```
   /path/to/janus/target/release/libjanus_ffi.so
   ```
   Or use the **Browse** button to locate it.

### 3.3 Assign Each Boundary

For each boundary marker object:

1. **Select** the boundary object in the viewport (e.g., `Boundary_West`).
2. In the Janus panel → **Setup** section:
   - Click **Auto-Assign Boundary Role** — the addon infers `west/east/south/north`
     based on the object's position relative to `JanusField`.
   - Or manually set **Boundary Role** from the dropdown.
3. Set **BC Type**: `DiffuseWall` (default), `Periodic`, `SpecularWall`, etc.
4. For `DiffuseWall`, set **Wall Temperature** (default: 300 K) and
   **Wall Velocity Ux/Uy** (default: 0).

> **Moving wall example**: Set `Boundary_North` to `DiffuseWall`,
> Temperature = 300 K, Wall Ux = 50 m/s → this creates a Couette flow.

### 3.4 Verify Role Assignment

After assignment, custom properties appear on each boundary object:

```python
import bpy
for role in ["west", "east", "south", "north"]:
    obj = bpy.data.objects.get(f"Boundary_{role.capitalize()}")
    assert obj is not None
    assert "janus_boundary_role" in obj
    assert obj["janus_boundary_role"] == role
    assert "janus_boundary_temperature" in obj
print("✓ Boundary roles verified")
```

---

## Stage 4: Configure Solver Parameters

### 4.1 Set Mesh and Output

In the Janus panel → **Simulate** section:

| Parameter | Field | Default | Notes |
|-----------|-------|---------|-------|
| **Field Mesh** | `mesh_object_name` | `JanusField` | Must match your grid object name |
| **Sim Output Dir** | `sim_output_dir` | `/home/.../janus/output` | Where `.jvtk` files are written |
| **Active Field** | `active_field` | `rho` | Field to display during simulation |

### 4.2 Choose Time Integrator

In the **Simulate** section, set **Integrator**:

| Setting | Enum | Description |
|---------|------|-------------|
| **Euler** | `0` | First-order, fastest per-step, most diffusive |
| **RK2** | `1` | Second-order Heun, balanced |
| **RK4** | `2` | Fourth-order, highest accuracy, ~same wall-clock cost* |

> *Benchmark result (32×32 grid): Euler = 139.5 ms/step, RK2 = 130.5 ms/step,
> RK4 = 126.2 ms/step. All schemes are dominated by the kinetic distribution
> evaluation (25² velocity points × 1024 cells), not the time-stepping overhead.
> **RK4 is recommended** — it provides 4th-order temporal accuracy at no
> meaningful extra cost on this problem size.

### 4.3 Set CFL Number and Steps per Tick

- **CFL**: `0.4` (default, safe for all schemes)
- **Steps / Tick**: `5` — solver advances this many steps per Blender timer tick
  (50 ms interval → up to 100 steps/second attempted)

---

## Stage 5: Setup Visualization Materials

### 5.1 Click "Setup Visualization Materials"

In the Janus panel → **Visualize** section:
1. Click **Setup Visualization Materials**.
2. The operator creates a **Principled BSDF** material with a **Color Ramp** node
   that maps field values to a blue→orange color gradient.
3. The viewport switches to **Material Preview** shading automatically.

### 5.2 Verify Material Creation

```python
import bpy
obj = bpy.data.objects["JanusField"]
mat = obj.data.materials[0]
assert mat.use_nodes
assert any(n.type == "BSDF_PRINCIPLED" for n in mat.node_tree.nodes)
assert any(n.type == "VALTORGB" for n in mat.node_tree.nodes)
print("✓ Material verified")
```

### 5.3 (Optional) Enable Regime Overlay

Toggle **Regime Overlay** in the Visualize section to apply a separate material
that colors cells by Knudsen number regime:

| Color | Regime | Kn range |
|-------|--------|----------|
| 🔵 Deep blue | Continuum | Kn < 0.01 |
| 🟢 Green | Slip | 0.01 ≤ Kn < 0.1 |
| 🟡 Yellow | Transition | 0.1 ≤ Kn < 10 |
| 🔴 Red | Free molecular | Kn ≥ 10 |

---

## Stage 6: Start the Simulation

### 6.1 Click "Start Simulation"

In the Janus panel → **Simulate** section, click **▶ Start Simulation**.

What happens internally:
1. The addon builds a case payload JSON from the scene (grid, BCs, gas properties).
2. `janus_solver_create()` is called via FFI — the solver initialises in Rust.
3. `janus_solver_cfl_dt()` computes the timestep: `dt = CFL × min(dx, dy) / v_max`.
4. An initial `.jvtk` snapshot is written to the output directory.
5. Blender's `bpy.app.timers` fires a callback every 50 ms that calls
   `janus_solver_step()` for `sim_steps_per_tick` iterations.

### 6.2 Monitor the Status Bar

The **Janus CFD** panel shows live diagnostics:

```
Status:       Running (rho)
Sim Time:     1.47e-06 s
Step:         300
```

These update on every timer tick (~20 Hz perceived refresh rate).

---

## Stage 7: Watch Live Field Updates

### 7.1 Viewport Coloring

After each timer tick, the field data is written to mesh face attributes:
- `rho`, `temperature`, `kn_loc`, `mom_x`, `mom_y`, `energy`

The material Color Ramp reads the active attribute and maps it to colors.
The viewport redraws automatically via `area.tag_redraw()`.

### 7.2 Switch the Active Field

While the simulation is running:
1. In the Janus panel, change **Field** to `temperature`, `kn_loc`, etc.
2. Click **Update Field Display** — the Color Ramp switches attribute source instantly.

### 7.3 Verify Field Data

```python
import bpy
mesh = bpy.data.objects["JanusField"].data
for attr in ["rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"]:
    assert attr in mesh.attributes, f"Missing attribute: {attr}"
    data = [d.value for d in mesh.attributes[attr].data]
    assert all(isinstance(v, float) for v in data)
print("✓ All field attributes populated")
```

---

## Stage 8: Stop Simulation & Export

### 8.1 Click "Stop Simulation"

Click **■ Stop Simulation** in the Janus panel. This:
1. Stops the Blender timer callback.
2. Calls `janus_solver_destroy()` to free Rust memory.
3. Writes the final `manifest.json` to the output directory.

### 8.2 Output Files

After stopping, the output directory contains:

```
output/
├── manifest.json          # Frame index: step, time, filename
├── live_0000.jvtk         # Initial state (t=0)
├── live_0001.jvtk         # Frame after write_every steps
├── live_0002.jvtk
└── ...
```

### 8.3 Read the .jvtk File (Python)

```python
import sys
sys.path.insert(0, "/path/to/janus/blender-addon")
from core.jvtk_reader import JvtkReader

reader = JvtkReader.open("output/live_0001.jvtk")
print(f"Grid: {reader.header.dims}, Time: {reader.header.time:.3e} s")
print(f"Available fields: {reader.available_fields()}")

rho = reader.cell_field("rho")
print(f"Density range: {rho.min():.4f} – {rho.max():.4f} kg/m³")
assert len(rho) == reader.header.ncells
assert all(r > 0 for r in rho), "Negative density detected!"
reader.close()
print("✓ .jvtk file verified")
```

### 8.4 Replay in Blender Timeline

1. Set **Data Directory** in the Janus panel to your output folder.
2. Set **Manifest** to the path of `manifest.json`.
3. Click **Import Manifest** — the frame list loads.
4. Drag the **Timeline** slider — the field mesh updates from `.jvtk` for each frame.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Janus FFI library not found" | Wrong library path | Set path in **Preferences → Add-ons → Janus CFD** |
| "CFL timestep is zero" | Grid too coarse or CFL=0 | Set CFL ≥ 0.1, verify `dx > 0` |
| Simulation starts but viewport doesn't update | Viewport shading ≠ Material | Switch viewport to **Material Preview** |
| `janus_solver_create` fails | Bad JSON payload | Check boundary tags have valid roles |
| Field colors are all the same | Color ramp range too wide | Use **Update Field Display** to refit |
| `.jvtk` file not created | Output directory is read-only | Set a writable **Sim Output Dir** |
| Very slow step time (>500 ms) | Large velocity grid (`n_per_dim`) | Reduce `n_per_dim` in case payload |

---

## Performance Reference

Measured on a 32×32 cell grid with 25² velocity points/cell:

| Integrator | Mean ms/step | Steps/s |
|-----------|-------------|---------|
| Euler     | ~140 ms     | ~7.2    |
| RK2       | ~131 ms     | ~7.7    |
| RK4       | ~126 ms     | ~7.9    |

> The dominant cost is the discrete velocity distribution evaluation (kinetic),
> not the time-integration scheme overhead.  All three integrators run at
> comparable wall-clock speed on a 32×32 grid.  **Use RK4** for maximum
> temporal accuracy at no meaningful extra cost.

Run the profiler yourself:
```bash
python3 scripts/benchmark_schemes.py --nx 32 --ny 32 --steps 50 --output bench.html
```

---

## Validation Commands

Run all automated tests to verify the installation:

```bash
# Unit + integration tests (no Blender needed)
python3 -m unittest discover -s blender-addon/tests -v

# CLI integration validation (10 stages)
python3 scripts/validate_workflow.py --verbose

# FFI performance benchmark
python3 scripts/benchmark_schemes.py --nx 32 --ny 32 --steps 30 --output bench.html
```

Expected output:
- **129 unit tests** — all pass
- **10/10 CLI stages** — all pass
- Benchmark exits with code 0, all schemes complete
