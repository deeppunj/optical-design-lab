# optical-design-lab
optical-design-lab/

├── README.md                 # what the project is, how to run it
├── LICENSE                   # MIT is a good portfolio default
├── .gitignore                # files Git must ignore
├── pyproject.toml            # project metadata and dependencies
├── docs/
│   └── lessons/              # theory notes, one file per stage
│       └── 01_paraxial_basics.md
├── src/
│   └── opticslab/
│       ├── __init__.py
│       ├── ray.py            # Ray (height, angle)
│       ├── elements.py       # Space, ThinLens, Aperture
│       ├── system.py         # OpticalSystem
│       └── analysis.py       # focal length, image position (later)
├── examples/
│   └── demo_single_lens.py   # runnable lesson scripts
├── tests/
│   └── test_paraxial.py      # physics sanity checks
└── ui/                       # PySide6 app (added at Stage 3)

