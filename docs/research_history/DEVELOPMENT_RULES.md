# Development Rules — Digital Media Verification

## 1. Git Safety

**All future development changes must happen on the `swaraj` branch.**

Before editing code:

```bash
git branch --show-current
```

Expected:

```text
swaraj
```

Do not make development commits on `main`.

`main` must remain the safe/stable branch.

## 2. Inspect Before Changing

Before modifying an existing component:

1. Read the current implementation.
2. Read related tests.
3. Identify the API/schema it depends on.
4. Make the smallest change that satisfies the requirement.
5. Run the relevant tests.

Do not replace working architecture without a demonstrated reason.

## 3. Preserve the Active Architecture

The active application is:

- React frontend
- FastAPI backend
- modular Python pipeline

The old Streamlit scaffolding is legacy.

Do not spend project time rebuilding Streamlit unless explicitly requested.

## 4. No Hallucinated Interfaces

Never invent fields, routes, model outputs, or configuration names.

When a change depends on the backend response:

- inspect the existing Pydantic schema;
- inspect the route;
- inspect the frontend API service;
- then update the frontend.

Keep frontend/backend contracts synchronized.

## 5. No Fake AI Outputs

Do not add placeholder results that look like real model output.

For demonstrations, static examples are allowed only when they are clearly labeled as demo fixtures and are not presented as live inference.

## 6. Model Integrity

Do not silently:

- change label mappings;
- swap model architectures;
- change checkpoint meaning;
- train on test data;
- tune on the final test set;
- report validation numbers as test performance.

Any experiment change must be documented.

## 7. GPU Verification

When GPU work is requested, verify:

- PyTorch CUDA availability;
- selected device;
- model placement;
- tensor placement;
- actual GPU utilization.

Use `nvidia-smi` while a representative workload is running.

Do not treat the presence of an NVIDIA driver as proof that PyTorch is using the GPU.

## 8. Explainability Integrity

Grad-CAM and token attribution must use the real model computation.

Do not draw decorative heatmaps or highlight arbitrary words just to improve the UI.

## 9. Uncertainty Integrity

Uncertainty values must come from the implemented MC Dropout/UQ process.

Do not rename confidence as uncertainty.

Do not describe the current thresholds as statistically calibrated unless calibration has actually been evaluated.

## 10. Frontend Design

Read `DESIGN.md` before making UI changes.

Use its tokens and principles consistently:

- warm cream canvas;
- warm near-black ink;
- restrained orange;
- white cards;
- hairlines;
- no shadows;
- editorial typography;
- JetBrains Mono on code/data surfaces.

Do not introduce random gradients, excessive glassmorphism, neon colors, or heavy shadows.

## 11. Testing Rule

After backend changes:

```bash
pytest -q
```

After frontend changes:

```bash
npm test -- --run
```

Use the project's actual package scripts if the command differs.

For training changes, additionally run a small smoke test before launching a full training job.

## 12. Documentation Rule

When behavior changes materially, update the relevant context/documentation.

Do not create duplicate contradictory architecture documents.

The intended source hierarchy is:

- `DESIGN.md` -> frontend visual/design rules
- `PROJECT_CONTEXT.md` -> project purpose, scope, responsible AI, overall constraints
- `ARCHITECTURE.md` -> technical architecture and pipeline behavior
- `docs/TRAINING_AND_EVALUATION.md` -> experiments, datasets, training, GPU, metrics
- `docs/DEMO_CONTEXT.md` -> final presentation/demo behavior
- `DEVELOPMENT_RULES.md` -> implementation and Git rules
