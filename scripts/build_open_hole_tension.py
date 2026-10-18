# Code under BSD-3-Clause (see LICENSE). (c) 2026 Francesco Mario Antonio Mitrotta.
"""
build_open_hole_tension.py
---------------------------
Builds, solves, and extracts results for a progressive-damage analysis of an open-hole
tension (OHT) plate: a square quasi-isotropic composite laminate with a central circular
hole, loaded in uniaxial in-plane tension until it fails. This reproduces the benchmark of
Liu et al. (2010) and Thapa et al. (2021), using Abaqus's built-in Hashin damage-initiation
criteria with an energy-based linear-softening damage evolution law -- unlike either
reference, which used strength-based criteria with no fracture energy (Thapa: sudden
stiffness knockdown to 1%; Liu: nodal-force release in a custom user element).

See notebooks/06_open_hole_tension.ipynb for the full theoretical background, every
parameter's citation, and the comparison against the two reference papers -- this script
intentionally keeps only short inline notes, not the full explanation.

Do not import this module, and do not run it directly inside a live Jupyter kernel: the
`from abaqus import *` line below immediately hands the whole script off to a real Abaqus
installation and terminates the calling process (see notebook 01, "Building the model in Abaqus", for why). Run it either
directly (`python build_open_hole_tension.py [key=value ...]`) or, from a notebook, via
`subprocess` so only the child process is affected.

Parameters (all optional, all `key=value` tokens, in any order):
    ncirc=<int>     elements along each quadrant's hole arc AND its opposite outer plate edge
                    (the circumferential direction of the radial "pie slice" mesh; default 60)
    nrad=<int>      elements along each radial cut, graded so elements are smallest at the
                    hole (default 45). The default 60 x 45 x 4 = 10,800-element mesh is NB6's
                    reference mesh, the coarsest one whose elements all respect the
                    matrix-tension element-size limit L_c < 1.33 mm.
    E1, E2, nu12, G12, G13, G23=<float>
                    lamina elastic constants, MPa (defaults: Liu Table 1 -- E1=172400,
                    E2=10300, nu12=0.32, G12=G13=5520, G23=3450)
    t_ply=<float>   ply thickness, mm (default 0.125, Liu p. 2372)
    X_T, X_C, Y_T, Y_C, S_L, S_T=<float>
                    Hashin strengths, MPa (defaults: Liu Table 1 -- X_T=2826.5, X_C=1620,
                    Y_T=65.5, Y_C=248, S_L=S_T=122)
    G_fT, G_fC, G_mT, G_mC=<float>
                    fracture energies, N/mm (defaults: Camanho, Maimi & Davila 2007, IM7-8552
                    -- 81.5, 106.3, 0.2774, 0.7879; a flagged resin substitution, see NB6)
    alpha=<float>   shear-influence coefficient in the tensile-fiber Hashin criterion (default 1)
    visc=<float>    damage-stabilization viscosity, all four modes (default 1e-4, chosen in NB6)
    U2=<float>      target displacement at the loaded (top) edge, mm (default 1.3, Thapa's)
    maxinc, mininc=<float>, maxnuminc=<int>
                    StaticStep automatic-incrementation settings (maxInc, minInc, maxNumInc;
                    the initial increment is set equal to maxInc). Abaqus's own defaults are
                    1 (no cap), 1e-5 and 100; this script's defaults, 0.01, 1e-9 and 5000, are
                    the values NB6 settles on, and NB6 explains why each differs. A run that
                    hits maxnuminc stops with every frame so far still in the .odb and is
                    extracted normally.
    cpus=<int>      cores for the solver (default 1), run as threads of one process
                    (multiprocessingMode=THREADS, set explicitly so Windows and Linux behave
                    alike), with numDomains set equal to cpus, which this kernel requires even
                    though the stub documents it for Abaqus/Explicit only. Abaqus refuses more
                    cores than the machine has.
    light=1         skip everything a campaign does not read back: the damage field output and
                    the node displacement output are not requested, and "_mesh.json",
                    "_damage.csv" and the per-frame histories in the JSON are not written. The
                    solver path, and so every number that is kept, is unchanged (default 0).
    mesh_only=1     build and mesh the part, write mesh statistics and the mesh itself, and stop
    inp_only=1      build the model, write the job's input file (<stem>.inp) and a JSON with the
                    mesh statistics, every element's Lc, the gross area and the ply angles, and stop
                    (the template of NB8's campaign, which solves it without CAE)
    out=<path>      output JSON path (default oht_results.json). The Abaqus job runs inside the
                    same directory, named after the JSON's stem. Companion files with the same
                    stem: "_mesh.json" (node coordinates and element connectivity, written
                    unless light=1) and "_damage.csv" (per-element, per-ply damage at the
                    tagged frames; full runs only).

Example:
    python build_open_hole_tension.py ncirc=20 nrad=15 out=../runs/nb06/M1.json

Requires: pip install "abqpy==2023.*" (match the version to your installed Abaqus).
"""
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------------------------
# Parameter parsing -- BEFORE the abaqus import, so it runs identically on both of
# abqpy's self-relaunch passes. This cannot be positional:
# Abaqus's noGUI launcher hands back its own full command line, appends anything after its
# own "--" at the very end, and rewrites any "name=value" it doesn't itself recognize into
# two separate tokens, "-name" then "value". The scan below handles both shapes.
# ---------------------------------------------------------------------------------
PARAMS = {
    "ncirc": "60",
    "nrad": "45",
    "e1": "172400.0",
    "e2": "10300.0",
    "nu12": "0.32",
    "g12": "5520.0",
    "g13": "5520.0",
    "g23": "3450.0",
    "t_ply": "0.125",
    "x_t": "2826.5",
    "x_c": "1620.0",
    "y_t": "65.5",
    "y_c": "248.0",
    "s_l": "122.0",
    "s_t": "122.0",
    "g_ft": "81.5",
    "g_fc": "106.3",
    "g_mt": "0.2774",
    "g_mc": "0.7879",
    "alpha": "1.0",
    "visc": "1e-4",
    "u2": "1.3",
    "maxinc": "0.01",
    "mininc": "1e-9",
    "maxnuminc": "5000",
    "cpus": "1",
    "light": "0",
    "mesh_only": "0",
    "inp_only": "0",
    "out": "oht_results.json",
}
argv = sys.argv
i = 0
while i < len(argv):
    token = argv[i]
    if "=" in token:
        key, _, value = token.partition("=")
        key = key.lstrip("-").strip().lower()
        if key in PARAMS:
            PARAMS[key] = value.strip()
        i += 1
    elif token.startswith("-") and token[1:].strip().lower() in PARAMS and i + 1 < len(argv):
        key = token[1:].strip().lower()
        PARAMS[key] = argv[i + 1].strip()
        i += 2
    else:
        i += 1

NCIRC = int(PARAMS["ncirc"])  # elements per quadrant, circumferential direction
NRAD = int(PARAMS["nrad"])  # elements per radial cut
E1 = float(PARAMS["e1"])  # MPa, fiber-direction modulus
E2 = float(PARAMS["e2"])  # MPa, transverse modulus
NU12 = float(PARAMS["nu12"])  # major Poisson's ratio
G12 = float(PARAMS["g12"])  # MPa, in-plane shear modulus
G13 = float(PARAMS["g13"])  # MPa, transverse shear modulus (shell transverse-shear stiffness)
G23 = float(PARAMS["g23"])  # MPa, transverse shear modulus (shell transverse-shear stiffness)
T_PLY = float(PARAMS["t_ply"])  # mm, ply thickness
X_T = float(PARAMS["x_t"])  # MPa
X_C = float(PARAMS["x_c"])  # MPa
Y_T = float(PARAMS["y_t"])  # MPa
Y_C = float(PARAMS["y_c"])  # MPa
S_L = float(PARAMS["s_l"])  # MPa
S_T = float(PARAMS["s_t"])  # MPa
G_FT = float(PARAMS["g_ft"])  # N/mm
G_FC = float(PARAMS["g_fc"])  # N/mm
G_MT = float(PARAMS["g_mt"])  # N/mm
G_MC = float(PARAMS["g_mc"])  # N/mm
ALPHA = float(PARAMS["alpha"])
VISC = float(PARAMS["visc"])  # damage-stabilization viscosity, in units of step time
TARGET_U2 = float(PARAMS["u2"])  # mm
MININC = float(PARAMS["mininc"])
MAXINC = float(PARAMS["maxinc"])
MAXNUMINC = int(PARAMS["maxnuminc"])
CPUS = int(PARAMS["cpus"])  # solver cores (threads of one process)
LIGHT = PARAMS["light"].strip() == "1"  # campaign mode: write only what a campaign reads back
MESH_ONLY = PARAMS["mesh_only"].strip() == "1"
INP_ONLY = PARAMS["inp_only"].strip() == "1"
# This script's folder, for importing extract_open_hole_tension.py. Under Abaqus/CAE noGUI the script
# path is one argv token (possibly "noGUI=<path>"), relative to the launch folder, so resolve it now.
SCRIPT_DIR = next(os.path.dirname(os.path.abspath(t.split("=")[-1])) for t in argv if t.endswith("build_open_hole_tension.py"))

# Resolve the output path to an ABSOLUTE path now, before the os.chdir() further down, so
# every named run (JSON, mesh JSON, damage CSV, and the Abaqus job's own files) is
# self-contained in one directory (normally runs/nb06/).
OUT_NAME = os.path.abspath(PARAMS["out"])
OUT_DIR = os.path.dirname(OUT_NAME)
if OUT_DIR and not os.path.isdir(OUT_DIR):  # Abaqus's kernel is Python 2: no exist_ok kwarg
    os.makedirs(OUT_DIR)
STEM = os.path.splitext(OUT_NAME)[0]
_stem = os.path.basename(STEM)
JOB_NAME = "".join(c if (c.isalnum() or c == "_") else "_" for c in _stem)[:60] or "OpenHoleTension"  # Abaqus job names reject ".", "-", etc.

from abaqus import *  # triggers the self-relaunch under real Abaqus -- see the module docstring
from abaqusConstants import *
import mesh  # mesh.ElemType
import section  # section.SectionLayer

# Epoch seconds of the script's own start. Taken after the abaqus import, which in abqpy's first
# pass relaunches the script under Abaqus: the gap between the driver's launch time and this
# stamp is the CAE start-up cost.
T_SCRIPT = time.time()

# ---------------------------------------------------------------------------------
# Geometry: a 76.2 mm square plate with a central 19.05 mm hole (Liu Fig. 1 / Thapa Fig. 3)
# ---------------------------------------------------------------------------------
L = 76.2  # mm, plate side length
D = 19.05  # mm, hole diameter
R = D / 2.0  # mm, hole radius
CX, CY = L / 2.0, L / 2.0  # hole is centered on the plate
R_CORNER = math.hypot(L / 2.0, L / 2.0)  # mm, distance from the plate center to any corner

# The four hole points the radial partition cuts start from (pointing straight at each corner)
corners = [(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0)]  # BL, BR, TR, TL
hole_pts = []
for ang_deg in (225.0, 315.0, 45.0, 135.0):  # matches the corner order above
    ang = math.radians(ang_deg)
    hole_pts.append((CX + R * math.cos(ang), CY + R * math.sin(ang), 0.0))

# The model is named after the run: Abaqus writes a temporary "_<model name>.aif" file into the
# job directory, so a fixed name makes concurrent runs sharing a directory collide.
MODEL_NAME = JOB_NAME
mdb.Model(name=MODEL_NAME)
model = mdb.models[MODEL_NAME]

sketch = model.ConstrainedSketch(name="plate_profile", sheetSize=200.0)
sketch.rectangle(point1=(0.0, 0.0), point2=(L, L))
sketch.CircleByCenterPerimeter(
    center=(CX, CY),
    point1=(hole_pts[0][0], hole_pts[0][1]),  # perimeter point at the SAME angle as the first
    # partition cut below -- a defining point at any other angle adds an extra seam vertex to
    # the circle, splitting one hole arc in two and making that region non-mappable
    # (structured meshing then silently produces zero elements).
)

part = model.Part(name="Plate", dimensionality=THREE_D, type=DEFORMABLE_BODY)
part.BaseShell(sketch=sketch)

# ---------------------------------------------------------------------------------
# Partition into 4 mappable "pie slice" regions -- the same topology as Liu's Fig. 2 and
# Thapa's Fig. 4: a shortest-path cut from each hole point out to the matching plate corner.
# Each face is then bounded by exactly 4 edges (hole arc, two radial cuts, one outer plate
# edge), which is what a STRUCTURED quad mesh requires.
# ---------------------------------------------------------------------------------
for k in range(4):
    faces = part.faces  # re-query every iteration: partitioning invalidates prior face handles
    target_corner = corners[k]
    nudge = 0.1  # mm; findAt needs a point ON the face to cut, nudged in from the corner
    dx, dy = CX - target_corner[0], CY - target_corner[1]
    norm = math.hypot(dx, dy)
    probe_pt = (target_corner[0] + nudge * dx / norm, target_corner[1] + nudge * dy / norm, 0.0)
    face_to_cut = faces.findAt((probe_pt,))
    part.PartitionFaceByShortestPath(faces=face_to_cut, point1=hole_pts[k], point2=target_corner)

# ---------------------------------------------------------------------------------
# Material: IM7/5250-4 (Liu et al. 2010, Table 1) with Hashin damage initiation,
# energy-based linear-softening evolution (Lapczyk & Hurtado 2007), and viscous damage
# stabilization (see NB6 for why a multi-element softening problem needs it).
# ---------------------------------------------------------------------------------
material = model.Material(name="IM7_5250-4")
material.Elastic(
    table=((E1, E2, NU12, G12, G13, G23),),  # [MPa], except nu12
    type=LAMINA,
)
material.HashinDamageInitiation(
    table=((X_T, X_C, Y_T, Y_C, S_L, S_T),),  # [MPa]
    alpha=ALPHA,  # NOTE: the abqpy stub also lists unrelated MSFLD parameters
)
material.hashinDamageInitiation.DamageEvolution(
    type=ENERGY,
    table=((G_FT, G_FC, G_MT, G_MC),),  # [N/mm]
    softening=LINEAR,
)
material.hashinDamageInitiation.DamageStabilization(  # missing from the abqpy stub; real
    fiberTensileCoeff=VISC,  # signature is four Float keywords
    fiberCompressiveCoeff=VISC,
    matrixTensileCoeff=VISC,
    matrixCompressiveCoeff=VISC,
)

# ---------------------------------------------------------------------------------
# Composite shell section: [45/0/-45/90]s, 8 plies of thickness t_ply, 3 Simpson points per ply.
# Ply angle convention: 0 deg = fibers along the load (global Y), Liu's Fig. 1. Abaqus's
# default material axis 1 for a flat shell in the XY plane is the projection of global X, so
# every physical ply angle is passed as (physical angle + 90 deg).
# ---------------------------------------------------------------------------------
ORIENT_OFFSET = 90.0  # degrees
physical_angles = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]  # ply 1 (top) to ply 8 (bottom)
ply_thickness = T_PLY  # mm

layers = []
for ply_index, angle in enumerate(physical_angles, start=1):
    layers.append(
        section.SectionLayer(
            thickness=ply_thickness,
            material="IM7_5250-4",
            orientAngle=angle + ORIENT_OFFSET,
            numIntPts=3,
            axis=AXIS_3,
            plyName="Ply-%d" % ply_index,
        )
    )
model.CompositeShellSection(name="Laminate", layup=layers, integrationRule=SIMPSON)
part.SectionAssignment(region=(part.faces,), sectionName="Laminate")

# ---------------------------------------------------------------------------------
# Mesh the part, THEN instance it (a dependent instance created before its part is meshed
# never picks up the mesh).
# ---------------------------------------------------------------------------------
part.setMeshControls(regions=part.faces, elemShape=QUAD, technique=STRUCTURED)
part.setElementType(
    regions=(part.faces,),
    elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),),
)

# Classify every edge by the distance of its two endpoints from the hole center: both ~R is
# a hole arc, both ~R_CORNER an outer plate edge, one of each a radial cut. For each radial
# cut, also record which end sits at the hole (vertices[i].pointOn is a NESTED single-point
# tuple on this install, not the flat 3-tuple the stub claims).
tol = 0.5  # mm
vertices = part.vertices
hole_edges, outer_edges, radial_hole_first, radial_hole_last = [], [], [], []
for e in part.edges:
    v_ids = e.getVertices()
    dists = []
    for vi in v_ids:
        vx, vy, vz = vertices[vi].pointOn[0]
        dists.append(math.hypot(vx - CX, vy - CY))
    d0, d1 = dists[0], dists[-1]
    if abs(d0 - R) < tol and abs(d1 - R) < tol:
        hole_edges.append(e)
    elif abs(d0 - R_CORNER) < tol and abs(d1 - R_CORNER) < tol:
        outer_edges.append(e)
    elif d0 < d1:
        radial_hole_first.append(e)  # hole end is the edge's first vertex
    else:
        radial_hole_last.append(e)

# Seeds. Hole arcs and outer edges: NCIRC uniform elements each. Radial cuts: NRAD elements,
# geometrically graded so the smallest sits at the hole, with a largest/smallest ratio equal
# to the outer-edge/hole-arc length ratio, 4L/(pi D) ~ 5.1 -- the circumferential element size
# grows by exactly that factor from the hole to the plate edge, so grading the radial size by
# the same factor keeps every element close to square, as in Liu's and Thapa's meshes (and as
# the crack-band length Lc = sqrt(area) assumes -- Lapczyk & Hurtado 2007, p. 2336).
# Per the stub, end1Edges puts the smallest elements at the edge's first vertex.
BIAS_RATIO = 4.0 * L / (math.pi * D)
part.seedEdgeByNumber(edges=hole_edges, number=NCIRC, constraint=FIXED)
part.seedEdgeByNumber(edges=outer_edges, number=NCIRC, constraint=FIXED)
seed_kwargs = {}
if radial_hole_first:
    seed_kwargs["end1Edges"] = radial_hole_first
if radial_hole_last:
    seed_kwargs["end2Edges"] = radial_hole_last
part.seedEdgeByBias(biasMethod=SINGLE, ratio=BIAS_RATIO, number=NRAD, constraint=FIXED, **seed_kwargs)

part.generateMesh()
N_ELEMENTS = len(part.elements)

assembly = model.rootAssembly
instance = assembly.Instance(name="Plate-1", part=part, dependent=ON)

# ---------------------------------------------------------------------------------
# Per-element area (shoelace formula), characteristic length Lc = sqrt(area), aspect ratio
# (longest / shortest edge) and undeformed centroid, measured from the actual mesh.
# `element.connectivity` holds 0-based SEQUENCE POSITIONS into instance.nodes, not labels.
# NOTE: `from abaqus import *` shadows the builtin `sum` -- accumulate manually.
# ---------------------------------------------------------------------------------
LC_CEILING_MT = 2.0 * G_MT * E2 / Y_T ** 2  # mm, matrix-tension snap-back limit (NB6)
lc_values, aspect_values = [], []
element_area, element_centroids = {}, {}
for element in instance.elements:
    coords = [instance.nodes[idx].coordinates for idx in element.connectivity]
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    area = 0.0
    edges_len = []
    for k in range(4):
        area += x[k] * y[(k + 1) % 4] - x[(k + 1) % 4] * y[k]
        edges_len.append(math.hypot(x[(k + 1) % 4] - x[k], y[(k + 1) % 4] - y[k]))
    area = abs(area) / 2.0
    element_area[element.label] = area
    lc_values.append(math.sqrt(area))
    aspect_values.append(max(edges_len) / min(edges_len))
    cx_total, cy_total = 0.0, 0.0
    for c in coords:
        cx_total += c[0]
        cy_total += c[1]
    element_centroids[element.label] = (cx_total / 4.0, cy_total / 4.0)
lc_sorted = sorted(lc_values)
n_over = 0
for v in lc_values:
    if v > LC_CEILING_MT:
        n_over += 1
mesh_stats = {
    "ncirc": NCIRC, "nrad": NRAD, "n_elements": N_ELEMENTS,
    "hole_edge_size_mm": math.pi * D / (4 * NCIRC),
    "lc_min": lc_sorted[0], "lc_max": lc_sorted[-1],
    "aspect_max": max(aspect_values),
    "frac_elements_over_ceiling": float(n_over) / N_ELEMENTS,
}

# The mesh itself, for the notebook's mesh and damage-map plots (a campaign does not plot it).
if not LIGHT:
    with open(STEM + "_mesh.json", "w") as f:
        json.dump(
            {
                "nodes": [[n.coordinates[0], n.coordinates[1]] for n in instance.nodes],
                "elements": [list(el.connectivity) for el in instance.elements],
                "labels": [el.label for el in instance.elements],
            },
            f,
        )

if MESH_ONLY:
    with open(OUT_NAME, "w") as f:
        json.dump(mesh_stats, f, indent=2)
    sys.exit(0)

# ---------------------------------------------------------------------------------
# Node sets, by position
# ---------------------------------------------------------------------------------
tol_n = 1e-3  # mm
big = 1e6  # mm
nodes = instance.nodes
top_nodes = nodes.getByBoundingBox(xMin=-big, xMax=big, yMin=L - tol_n, yMax=L + tol_n, zMin=-big, zMax=big)
bottom_nodes = nodes.getByBoundingBox(xMin=-big, xMax=big, yMin=-tol_n, yMax=tol_n, zMin=-big, zMax=big)
# The single bottom-edge node nearest the centerline x=L/2 (no node need sit exactly there),
# re-queried by bounding box because assembly.Set needs a MeshNodeArray, not a Python tuple.
nearest_x = min(bottom_nodes, key=lambda n: abs(n.coordinates[0] - L / 2.0)).coordinates[0]
bottom_center_nodes = nodes.getByBoundingBox(
    xMin=nearest_x - tol_n, xMax=nearest_x + tol_n, yMin=-tol_n, yMax=tol_n, zMin=-big, zMax=big,
)
assembly.Set(name="Top-Edge", nodes=top_nodes)
assembly.Set(name="Bottom-Edge", nodes=bottom_nodes)
assembly.Set(name="Bottom-Center", nodes=bottom_center_nodes)

# ---------------------------------------------------------------------------------
# Step: Static, General, displacement-controlled, geometrically nonlinear. Incrementation
# settings are CLI parameters; NB6 explains each departure from Abaqus's own defaults.
# ---------------------------------------------------------------------------------
model.StaticStep(
    name="Tension",
    previous="Initial",
    nlgeom=ON,
    timePeriod=1.0,  # step time 0->1 maps linearly onto U2 = 0->TARGET_U2 (default Ramp amplitude)
    initialInc=MAXINC,  # start at the cap (Abaqus rejects initialInc > maxInc)
    minInc=MININC,
    maxInc=MAXINC,
    maxNumInc=MAXNUMINC,
)

# ---------------------------------------------------------------------------------
# Boundary conditions -- Liu's scheme (rollers on the fixed edge), not Thapa's clamped edge.
# U3/UR1/UR2 are fixed at ONE node only: the symmetric layup has B = 0, so the exact solution
# of this in-plane problem has no out-of-plane deflection or rotation anywhere; these three
# DOFs only remove the out-of-plane rigid-body modes.
# ---------------------------------------------------------------------------------
model.DisplacementBC(name="U2-fixed", createStepName="Initial", region=assembly.sets["Bottom-Edge"], u2=SET)
model.DisplacementBC(name="U1-fixed-center", createStepName="Initial", region=assembly.sets["Bottom-Center"], u1=SET)
model.DisplacementBC(
    name="Membrane-constraint", createStepName="Initial", region=assembly.sets["Bottom-Center"],
    u3=SET, ur1=SET, ur2=SET,
)
model.DisplacementBC(name="Applied-Tension", createStepName="Tension", region=assembly.sets["Top-Edge"], u2=TARGET_U2)

# ---------------------------------------------------------------------------------
# Output requests, edited into the keyword block (this install's Model has no output-request
# API). Element output at the MIDDLE section point of each ply only (2, 5,
# ..., 23): the state is pure membrane, so a ply's three section points carry the same value.
# Node output U only, to view the deformed shape in Abaqus/CAE; the notebook reads none.
# ---------------------------------------------------------------------------------
model.keywordBlock.synchVersions(storeNodesAndElements=False)
sie_blocks = model.keywordBlock.sieBlocks
field_output_index = next(i for i, line in enumerate(sie_blocks) if line.strip().startswith("*Output, field, variable=PRESELECT"))
# light=1: the Hashin criteria are all first-ply failure and the first 0-deg failure need; the
# damage variables (maps, per-ply maxima) and the nodal displacement are not read back.
element_variables = "HSNFTCRT, HSNFCCRT, HSNMTCRT, HSNMCCRT"
node_output = ""
if not LIGHT:
    element_variables = "DAMAGEFT, DAMAGEFC, DAMAGEMT, DAMAGEMC, " + element_variables
    node_output = "\n*Node Output\nU"
model.keywordBlock.replace(
    field_output_index,
    "*Output, field\n"
    "*Element Output\n"
    "2, 5, 8, 11, 14, 17, 20, 23\n"
    + element_variables
    + node_output,
)
sie_blocks = model.keywordBlock.sieBlocks
history_output_index = next(i for i, line in enumerate(sie_blocks) if line.strip().startswith("*Output, history, variable=PRESELECT"))
model.keywordBlock.replace(
    history_output_index,
    "*Output, history\n"
    "*Node Output, nset=Top-Edge\n"
    "U2, RF2\n"
    "*Energy Output\n"
    "ALLCD, ALLDMD",  # viscous vs. damage dissipation (NB6)
)

# ---------------------------------------------------------------------------------
# Job: submit and wait. chdir into OUT_DIR so every job file lands next to its own JSON.
# CPUS cores as threads of one process: made explicit because this install's site default is MPI
# (its site environment files switch to MPI when Microsoft MPI is installed), and a campaign must
# behave the same on Windows and Linux. numDomains must
# equal numCpus: the stub documents it for Abaqus/Explicit only, but this kernel refuses
# cpus > 1 with the default of 1 domain ("Number of domains must be a multiple of the number of
# processors"), even in threads mode.
# ---------------------------------------------------------------------------------
os.chdir(OUT_DIR)
job = mdb.Job(name=JOB_NAME, model=MODEL_NAME, numCpus=CPUS, numDomains=CPUS, multiprocessingMode=THREADS)
LAMINATE_THICKNESS = 8 * ply_thickness  # mm
GROSS_AREA = L * LAMINATE_THICKNESS  # mm^2, Liu's gross section (full width x thickness)
PARAMS_ECHO = {
    "run": _stem,
    "e1": E1, "e2": E2, "nu12": NU12, "g12": G12, "g13": G13, "g23": G23, "t_ply": T_PLY,
    "x_t": X_T, "x_c": X_C, "y_t": Y_T, "y_c": Y_C, "s_l": S_L, "s_t": S_T,
    "g_ft": G_FT, "g_fc": G_FC, "g_mt": G_MT, "g_mc": G_MC,
    "alpha": ALPHA, "visc": VISC, "u2_target": TARGET_U2,
    "mininc": MININC, "maxinc": MAXINC, "maxnuminc": MAXNUMINC,
    "cpus": CPUS, "light": LIGHT,
}

# inp_only=1: write the input file (<job>.inp, next to the JSON) and stop. A campaign patches its
# material data lines per sample and solves it without CAE (oht_campaign.py, NB8); the JSON carries
# what that needs: every element's Lc for the sample's own ceiling, the gross area and the ply angles.
if INP_ONLY:
    job.writeInput(consistencyChecking=OFF)
    meta = dict(mesh_stats)
    meta.update(PARAMS_ECHO)
    meta.update({"lc_values": lc_values, "gross_area_mm2": GROSS_AREA, "physical_angles": physical_angles})
    with open(OUT_NAME, "w") as f:
        json.dump(meta, f, indent=2)
    sys.exit(0)

T_SUBMIT = time.time()  # the model is built; the solver licence is requested from here
job.submit()
job.waitForCompletion()  # returns on success AND on failure -- check the .odb, not this
T_SOLVED = time.time()

# ---------------------------------------------------------------------------------
# Extraction: extract_open_hole_tension.py, shared with campaigns that solve without CAE.
# ---------------------------------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)
from extract_open_hole_tension import extract

extracted = extract(JOB_NAME, LIGHT, GROSS_AREA, physical_angles, element_centroids,
                    None if LIGHT else STEM + "_damage.csv")

# ---------------------------------------------------------------------------------
# Main results JSON -- every CLI parameter echoed back, plus everything NB6 reads.
# ---------------------------------------------------------------------------------
results = dict(mesh_stats)
results.update(PARAMS_ECHO)
results.update({"t_script": T_SCRIPT, "t_submit": T_SUBMIT, "t_solved": T_SOLVED})  # epoch s
results.update(extracted)
results["t_extracted"] = time.time()
with open(OUT_NAME, "w") as f:
    json.dump(results, f, indent=2)
