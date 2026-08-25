# System architecture (editable mermaid source)

```mermaid
flowchart LR
    GEN["Scene generator<br/>procedural cities (v2 layouts)<br/>disaster fates · casualty placement"]
    SIM["2.5D disaster sim<br/>heightfield, 2 m cells<br/>drones at real altitude<br/>3D ray-marched line of sight"]
    RF["Simulated RayFronts<br/>voxels · rays · frontiers<br/>open-set embeddings<br/>persistent semantic map"]
    BEL["Per-drone belief<br/>own sensing + gossip<br/>range/relay comms<br/>rays · coverage · segments · visited"]
    TOK["Token builder<br/>frontier · ray · segment · visited<br/>peer tokens + query tokens"]
    POL["TokenPolicy<br/>transformer ≈ 0.4 M params<br/>runs on every drone<br/>→ next target"]
    LLM["LLM query controller (Claude)<br/>disaster context + live map digest<br/>→ add / remove / reweight hint queries"]
    TRN["CTDE training<br/>DAgger from privileged oracle → MAPPO<br/>central critic (training only)<br/>reward: finds − time − redundancy − confirmed revisits"]

    GEN --> SIM --> RF --> BEL --> TOK --> POL
    BEL -. live semantics digest .-> LLM
    LLM -. query edits .-> TOK
    TRN -. gradients .-> POL

    style LLM fill:#f6e9fb,stroke:#a020c0
    style TRN fill:#e8eef8,stroke:#2f6fdb
    style POL fill:#e4f0f0,stroke:#00a0a0
```

Key points the figure should carry on a slide:
- **Decentralized execution**: everything right of RayFronts runs per drone; there is no central planner.
- **Open-set semantics**: the policy consumes raw embeddings + query tokens, never class labels or thresholds.
- **The LLM is out of the control loop**: it edits the query channel closed-loop from the live map digest; the policy decides where to fly.
- **CTDE**: the central critic exists only during training.
