"""
build_open_hole_tension.py
---------------------------
Builds, solves, and extracts results for a progressive-damage analysis of an open-hole
tension (OHT) plate: a square quasi-isotropic composite laminate with a central circular
hole, loaded in uniaxial in-plane tension until it fails. This reproduces the benchmark of
Thapa et al. (2021), itself a reproduction of the experiment and mesh-sensitivity study of
Liu et al. (2010), using Abaqus's built-in Hashin damage-initiation criteria with an
energy-based linear-softening damage evolution law -- unlike either reference, which used
strength-based criteria with no fracture energy and no mesh-objective length scale (Thapa:
sudden stiffness knockdown to 1%; Liu: nodal-force release in a custom user element).

See notebooks/06_open_hole_tension.ipynb for the full theoretical background, every
parameter's citation, and the comparison against the two reference papers -- this script
intentionally keeps only short inline notes, not the full explanation.

This script is independent of the notebook and accepts its parameters on the command line
as `key=value` tokens (see "Parameters" below) -- it is never written to disk by a notebook
cell.

Do not import this module, and do not run it directly inside a live Jupyter kernel: the
`from abaqus import *` line below immediately hands the whole script off to a real Abaqus
installation and terminates the calling process (see CLAUDE.md for why). Run it either
directly (`python build_open_hole_tension.py [key=value ...]`) or, from a notebook, via
`subprocess` so only the child process is affected.

Parameters (all optional, all `key=value` tokens, in any order):
    nhole=<int>     elements around the FULL hole circumference -- the mesh-density knob;
                    the four "pie slice" mesh regions (see "Mesh" below) each get nhole/4
                    elements along their hole arc and matching outer edge, and a radial
                    element count chosen so elements near the hole are roughly square
                    (default 60, giving ~1.0 mm elements at the hole edge -- see the
                    notebook's Part 4 for why this, not Thapa's 1200-element mesh, is the
                    finest mesh this material's fracture energies actually allow)
    nlgeom=ON|OFF   geometric nonlinearity for the Static step (default ON)
    X_T=<float>     fiber (longitudinal) tensile strength, MPa (default 2826.5, Liu Table 1
                    -- NOT Thapa's 2526.5, almost certainly a transcription slip; see NB6)
    X_C=<float>     fiber compressive strength, MPa (default 1620.0, Liu Table 1)
    Y_T=<float>     matrix (transverse) tensile strength, MPa (default 65.5, Liu Table 1)
    Y_C=<float>     matrix compressive strength, MPa (default 248.0, Liu Table 1)
    S_L=<float>     longitudinal shear strength, MPa (default 122.0, Liu Table 1)
    S_T=<float>     transverse shear strength, MPa (default 122.0, Liu Table 1)
    G_fT=<float>    fiber-tension fracture energy, N/mm (default 81.5 -- Camanho, Maimi &
                    Davila 2007, Table 5, G_1+ for IM7-8552; a flagged resin substitution,
                    see NB6 -- no published value exists for IM7/5250-4)
    G_fC=<float>    fiber-compression fracture energy, N/mm (default 106.3, same source,
                    Table 5, G_1-)
    G_mT=<float>    matrix-tension fracture energy, N/mm (default 0.2774, same source,
                    Table 3, G_2+)
    G_mC=<float>    matrix-compression fracture energy, N/mm (default 0.7879, same source,
                    Table 3, G_6 -- Abaqus's matrix-compression mode is shear-driven and no
                    separate G_2- was measured, so it is fed the mode-II shear energy; see NB6)
    visc=<float>    damage-stabilization viscosity, applied identically to all four Hashin
                    modes (default 1e-2 -- NOT the NB1-single-element scale of 1e-4, which
                    this notebook's own first attempt found never gets this multi-element
                    model's solver past the onset of softening at all; see NB6 Part 5 for
                    what this buys and how its cost is checked against the run's
                    ALLIE/ALLCD energy history)
    U2=<float>      target displacement at the loaded (top) edge, mm (default 1.3, Thapa's
                    value)
    out=<path>      output JSON path, resolved against the current working directory if
                    relative (default oht_results.json); any parent directories are created
                    if they don't already exist. A companion CSV of per-element, per-ply
                    damage at two tagged frames is written alongside it (same stem, with a
                    "_damage.csv" suffix) for the notebook's damage maps.

Example:
    python build_open_hole_tension.py nhole=60 U2=1.3 out=../runs/nb06/oht_reference.json

Requires: pip install "abqpy==2023.*" (match the version to your installed Abaqus).
"""
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------------------------
# Parameter parsing -- BEFORE the abaqus import, so it runs identically on both of
# abqpy's self-relaunch passes. See build_fiber_tension.py's own module docstring and
# abqpy_notes.md for the full story of why this cannot be positional (sys.argv[1], ...):
# Abaqus's noGUI launcher hands back its own full command line, appends anything after its
# own "--" at the very end, and rewrites any "name=value" it doesn't itself recognize into
# two separate tokens, "-name" then "value". The scan below handles both shapes.
# ---------------------------------------------------------------------------------
PARAMS = {
    "nhole": "60",
    "nlgeom": "ON",
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
    "visc": "1e-2",
    "u2": "1.3",
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

NHOLE = int(PARAMS["nhole"])  # elements around the full hole circumference
NLGEOM_ON = PARAMS["nlgeom"].upper() != "OFF"  # anything but a literal "OFF" is treated as ON
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
VISC = float(PARAMS["visc"])  # damage-stabilization viscosity, same units as step time
TARGET_U2 = float(PARAMS["u2"])  # mm
OUT_NAME = PARAMS["out"]

from abaqus import *  # triggers the self-relaunch under real Abaqus -- see the module docstring
from abaqusConstants import *  # symbolic constants used throughout (ON, OFF, LAMINA, S4R, ...)
import mesh  # mesh.ElemType, used below to force the S4R element choice
import section  # section.SectionLayer, used below to build the composite shell layup

# ---------------------------------------------------------------------------------
# Geometry: a 76.2 mm square plate with a central 19.05 mm hole (Thapa Fig. 3 / Liu Fig. 1)
# ---------------------------------------------------------------------------------
L = 76.2  # mm, plate side length
D = 19.05  # mm, hole diameter
R = D / 2.0  # mm, hole radius
CX, CY = L / 2.0, L / 2.0  # hole is centered on the plate
R_CORNER = math.hypot(L / 2.0, L / 2.0)  # mm, distance from the plate center to any corner

# The four hole points the radial partition cuts start from (at 45 deg to the plate edges,
# i.e. pointing straight at each corner) -- computed before the sketch is built so the
# circle's own defining point can be made to coincide with hole_pts[0] (see the comment
# by the CircleByCenterPerimeter call below for why this matters).
corners = [(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0)]  # BL, BR, TR, TL
hole_pts = []
for ang_deg in (225.0, 315.0, 45.0, 135.0):  # matches the corner order above
    ang = math.radians(ang_deg)
    hole_pts.append((CX + R * math.cos(ang), CY + R * math.sin(ang), 0.0))

mdb.Model(name="OpenHoleTension")  # create a new, empty model database entry
model = mdb.models["OpenHoleTension"]  # keep a handle to it for everything that follows

sketch = model.ConstrainedSketch(name="plate_profile", sheetSize=200.0)  # 2D sketch, canvas well over 76.2 mm
sketch.rectangle(point1=(0.0, 0.0), point2=(L, L))  # the 76.2x76.2 mm plate outline
sketch.CircleByCenterPerimeter(
    center=(CX, CY),
    point1=(hole_pts[0][0], hole_pts[0][1]),  # perimeter point at the SAME angle as the first
    # partition cut below (225 deg), not the more natural-looking (CX+R, CY) -- confirmed live
    # that leaving the circle's own defining point at a different angle than every partition
    # cut adds an extra seam vertex to the circle, splitting one of the four hole arcs into
    # two after partitioning (5 hole edges instead of 4) and making the resulting region
    # non-mappable (structured meshing then silently produces zero elements).
)

part = model.Part(name="Plate", dimensionality=THREE_D, type=DEFORMABLE_BODY)  # an empty deformable part
part.BaseShell(sketch=sketch)  # extrude the sketch (square minus circle) into a shell

# ---------------------------------------------------------------------------------
# Partition into 4 mappable "pie slice" regions: a shortest-path cut from each of the 4
# hole points above out to the matching plate corner. Each resulting face is bounded by
# exactly 4 edges -- one hole arc, two straight radial cuts, and one full outer plate edge
# -- which is what a STRUCTURED/QUAD mesh technique requires (see NB6 Part 6).
# ---------------------------------------------------------------------------------
for k in range(4):
    faces = part.faces  # re-query every iteration: partitioning invalidates prior face handles
    target_corner = corners[k]
    # findAt needs a point ON the face to cut, not just near it -- nudge a small distance in
    # from the corner, along the corner-to-center direction, to land reliably inside the face
    nudge = 0.1  # mm
    dx, dy = CX - target_corner[0], CY - target_corner[1]
    norm = math.hypot(dx, dy)
    probe_pt = (target_corner[0] + nudge * dx / norm, target_corner[1] + nudge * dy / norm, 0.0)
    face_to_cut = faces.findAt((probe_pt,))
    part.PartitionFaceByShortestPath(faces=face_to_cut, point1=hole_pts[k], point2=target_corner)

# ---------------------------------------------------------------------------------
# Material: IM7/5250-4 carbon/epoxy (Liu et al. 2010, Table 1) with Hashin damage
# initiation and energy-based linear-softening damage evolution (Lapczyk & Hurtado 2007),
# plus viscous damage stabilization -- see NB6 Parts 4-5 for why a multi-element softening
# problem needs the latter when NB1's single element did not.
# ---------------------------------------------------------------------------------
material = model.Material(name="IM7_5250-4")  # create the material definition

material.Elastic(
    table=((172400.0, 10300.0, 0.32, 5520.0, 5520.0, 3450.0),),  # E1, E2, nu12, G12, G13, G23 [MPa]
    type=LAMINA,  # orthotropic engineering-constants form
)

material.HashinDamageInitiation(
    table=((X_T, X_C, Y_T, Y_C, S_L, S_T),),  # XT, XC, YT, YC, SL, ST [MPa] -- all CLI parameters above
    alpha=1.0,  # shear-influence coefficient in the tensile fiber criterion, admitting shear
    # stress into that mode -- Hashin's own original (1980) formulation, and the same choice
    # NB1's build_fiber_tension.py makes, so this script stays consistent with the rest of the
    # series. A first attempt at alpha=0 (excluding shear entirely) converged but overshot
    # every reference number by 30-40%: with the tensile fiber mode harder to trigger, the
    # +-45 plies never picked up the spurious-looking-but-load-bearing damage Liu's own Hashin
    # column also reports (see NB6 Part 9 for exactly this comparison, and Part 7 for the
    # numbers this alpha=0 run actually produced).
    # NOTE: abqpy's local type stub for this method also lists several MSFLD-criterion
    # parameters (feq, fnn, frequency, ...) that do not apply to Hashin -- a known
    # documentation-generation bug, not a real Hashin parameter. See abqpy_notes.md.
)

material.hashinDamageInitiation.DamageEvolution(
    type=ENERGY,  # the evolution law is driven by fracture energy, not a displacement value directly
    table=((G_FT, G_FC, G_MT, G_MC),),  # GfT, GfC, GmT, GmC [N/mm] -- all CLI parameters above
    softening=LINEAR,  # stress drops linearly with equivalent displacement once damage initiates
)

material.hashinDamageInitiation.DamageStabilization(
    fiberTensileCoeff=VISC,  # NOTE: plain (non-cohesive) DamageStabilization is MISSING from
    fiberCompressiveCoeff=VISC,  # the abqpy stub entirely -- only DamageStabilizationCohesive
    matrixTensileCoeff=VISC,  # is present (confirmed live: grepping the installed stub for
    matrixCompressiveCoeff=VISC,  # "def DamageStabilization" finds only the cohesive variant).
    # The real method exists and works on this install; it just has no autocomplete or type
    # hint. Its real signature is four separate required Float keywords, not a table -- see
    # abqpy_notes.md. A single scalar viscosity (the CLI parameter above) is applied to all
    # four Hashin modes here, rather than tuning each independently, as the simplest choice
    # that still lets NB6 check its cost via the ALLIE/ALLCD energy ratio.
)

# ---------------------------------------------------------------------------------
# Composite shell section: [45/0/-45/90]s, 8 plies of 0.125 mm each (Liu Table 1 / Thapa
# Fig. 3), 3 Simpson integration points per ply -- see NB6 Part 2 for what a section point
# is and why 3 per ply is enough to resolve linear through-ply bending of stress.
#
# Ply angle convention: 0 deg means the ply's fibers run ALONG THE LOAD (global Y), matching
# Liu's Fig. 1 convention (which Thapa also uses for its own validation run). Abaqus's
# DEFAULT material orientation for a flat shell lying in the global XY plane has local
# material axis 1 aligned with the PROJECTION of global X onto the shell, not global Y --
# so every physical ply angle below is passed to Abaqus as (physical angle + 90 deg) rather
# than adding a separate MaterialOrientation datum coordinate system, which needs no
# additional geometry and is checked directly against the run: the physical-90-deg ply
# (fibers along global X, transverse to the load) is the one this notebook's Part 7 expects
# to show the earliest and largest DAMAGEMT, matching Liu's own finding that matrix cracking
# initiates in "the 90 deg ply" under their identical angle convention.
# ---------------------------------------------------------------------------------
ORIENT_OFFSET = 90.0  # degrees; see the comment above
physical_angles = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]  # [45/0/-45/90]s, ply 1 (top) to ply 8 (bottom)
ply_thickness = 0.125  # mm, per ply (Liu Table 1)

layers = []
for ply_index, angle in enumerate(physical_angles, start=1):
    layers.append(
        section.SectionLayer(
            thickness=ply_thickness,
            material="IM7_5250-4",
            orientAngle=angle + ORIENT_OFFSET,  # physical angle, shifted into Abaqus's default frame
            numIntPts=3,
            axis=AXIS_3,  # rotate the ply angle about the shell normal (the only sensible axis for a shell)
            plyName="Ply-%d" % ply_index,  # lets the notebook find a ply's section points by name later
        )
    )

model.CompositeShellSection(
    name="Laminate",
    layup=layers,  # all 8 layers given explicitly (not `symmetric=ON` mirroring 4) -- simpler
    # and unambiguous to read back later, at the cost of repeating the 4 numbers once
    integrationRule=SIMPSON,
)
part.SectionAssignment(region=(part.faces,), sectionName="Laminate")  # apply the laminate to every partitioned face

# ---------------------------------------------------------------------------------
# Mesh the part, THEN instance it: force a structured quad mesh on every "pie slice" region.
# See abqpy_notes.md -- a dependent instance created before its part is meshed does not pick
# up the mesh afterward, so the part must be meshed first.
# ---------------------------------------------------------------------------------
part.setMeshControls(regions=part.faces, elemShape=QUAD, technique=STRUCTURED)  # every region must be mappable for this to succeed
part.setElementType(
    regions=(part.faces,),
    elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),),  # 4-node reduced-integration shell, as in NB1
)

# Classify every edge by the distance of its TWO ENDPOINTS from the hole center (not a
# single sample point, which can misclassify a straight edge whose midpoint happens to sit
# near the hole radius): both ~= R is a hole arc, both ~= R_CORNER is an outer plate edge,
# one of each is a radial cut line. Confirmed live against this install's real vertex
# geometry (part.vertices[i].pointOn returns a NESTED single-point tuple, ((x, y, z),), like
# Edge.pointOn -- not the flat 3-tuple the local stub's type hint claims).
tol = 0.5  # mm
vertices = part.vertices
hole_edges, outer_edges, radial_edges = [], [], []
for e in part.edges:
    dists = []
    for vi in e.getVertices():
        vx, vy, vz = vertices[vi].pointOn[0]
        dists.append(math.hypot(vx - CX, vy - CY))
    if len(dists) == 1:
        dists = dists * 2  # a closed-curve edge has only one distinct vertex; shouldn't occur here
    d0, d1 = dists[0], dists[1]
    if abs(d0 - R) < tol and abs(d1 - R) < tol:
        hole_edges.append(e)
    elif abs(d0 - R_CORNER) < tol and abs(d1 - R_CORNER) < tol:
        outer_edges.append(e)
    else:
        radial_edges.append(e)  # one endpoint near R, the other near R_CORNER

# Seed: NHOLE elements around the full hole circumference -> NHOLE/4 per quadrant arc (the
# circumferential direction, shared between each hole arc and its opposite outer plate edge
# in a mapped "pie slice"); radial direction sized so elements near the hole are roughly
# square. This gives a UNIFORM element size across the whole plate (no grading away from
# the hole) -- a deliberate simplification for a first version of this script, flagged here
# rather than fixed, since it is what makes the reference mesh (NHOLE=60) noticeably more
# expensive than it needs to be far from the hole, which if anything only strengthens NB7's
# motivation for a surrogate.
n_circ = max(1, NHOLE // 4)  # elements along each quadrant's hole arc and outer edge
elem_size_at_hole = (2 * math.pi * R) / (n_circ * 4)  # mm; the actual element size this rounds to
radial_len = R_CORNER - R  # mm, hole edge to plate corner along a radial cut line
n_radial = max(1, int(round(radial_len / elem_size_at_hole)))  # elements along each radial cut line

part.seedEdgeByNumber(edges=hole_edges, number=n_circ, constraint=FIXED)
part.seedEdgeByNumber(edges=outer_edges, number=n_circ, constraint=FIXED)
part.seedEdgeByNumber(edges=radial_edges, number=n_radial, constraint=FIXED)

part.generateMesh()  # actually mesh the part with the controls and seeds set above
N_ELEMENTS = len(part.elements)  # recorded in the output JSON below

assembly = model.rootAssembly  # the (initially empty) assembly that instances live in
instance = assembly.Instance(name="Plate-1", part=part, dependent=ON)  # place the meshed part; "dependent" shares its mesh

# ---------------------------------------------------------------------------------
# Node sets, defined by position (robust to whatever node numbering the mesher assigns)
# ---------------------------------------------------------------------------------
tol_n = 1e-3  # mm; tolerance for the bounding-box node searches below
big = 1e6  # mm; a bound far outside the model, to make a bounding box unbounded in one direction
nodes = instance.nodes
top_nodes = nodes.getByBoundingBox(xMin=-big, xMax=big, yMin=L - tol_n, yMax=L + tol_n, zMin=-big, zMax=big)  # y=L, the loaded edge
bottom_nodes = nodes.getByBoundingBox(xMin=-big, xMax=big, yMin=-tol_n, yMax=tol_n, zMin=-big, zMax=big)  # y=0, the fixed edge
# the single bottom-edge node nearest the plate's centerline (x=L/2), which carries the one
# extra U1 constraint needed to remove rigid-body translation in x -- see NB6 Part 6 for why
# this node, and not some other, keeps the constraint from breaking left-right symmetry.
# A tight bounding box around x=L/2 finds nothing: the outer-edge node spacing (L / n_circ)
# does not generally divide L/2 evenly, so no node sits exactly there. Scan bottom_nodes in
# Python instead and keep whichever one is closest.
nearest_node = min(bottom_nodes, key=lambda n: abs(n.coordinates[0] - L / 2.0))
nearest_x = nearest_node.coordinates[0]
bottom_center_nodes = nodes.getByBoundingBox(  # re-query by that exact x, rather than passing a
    xMin=nearest_x - tol_n, xMax=nearest_x + tol_n,  # plain Python tuple of Node objects -- assembly.Set
    yMin=-tol_n, yMax=tol_n, zMin=-big, zMax=big,  # needs a proper MeshNodeArray, which only
)  # getByBoundingBox (not a Python min()/tuple) returns

assembly.Set(name="Top-Edge", nodes=top_nodes)
assembly.Set(name="Bottom-Edge", nodes=bottom_nodes)
assembly.Set(name="Bottom-Center", nodes=bottom_center_nodes)
assembly.Set(name="All-Nodes", nodes=instance.nodes)  # the whole plate, for the U3/UR1/UR2 membrane constraint below

# ---------------------------------------------------------------------------------
# Step: Static, General (displacement-controlled), with automatic incrementation tuned for
# a softening problem -- much smaller minInc and a much larger maxNumInc than NB1's single-
# element step, since softening forces many more, smaller increments before convergence.
# ---------------------------------------------------------------------------------
model.StaticStep(
    name="Tension",
    previous="Initial",
    nlgeom=ON if NLGEOM_ON else OFF,
    timePeriod=1.0,  # step runs 0->1; with the default Ramp amplitude this maps linearly onto 0->TARGET_U2
    initialInc=0.01,  # increment size to attempt first
    minInc=1e-8,  # allow much smaller increments than NB1's single element ever needed
    maxNumInc=10000,  # a softening solve can need thousands of small increments to converge
)

# ---------------------------------------------------------------------------------
# Boundary conditions -- Liu's scheme (rollers on the fixed edge), not Thapa's (a fully
# clamped edge with a rigid tie on the loaded edge, which suppresses the Poisson contraction
# CLAUDE.md's boundary-condition rule warns against). See NB6 Part 6 for the full membrane
# argument for why U3/UR1/UR2 can be fixed everywhere at no physical cost for this symmetric,
# in-plane-loaded laminate (B=0, so the exact solution already has zero bending and zero
# out-of-plane rotation -- a claim this script's own reaction-force output lets NB6 verify).
# ---------------------------------------------------------------------------------
model.DisplacementBC(name="U2-fixed", createStepName="Initial", region=assembly.sets["Bottom-Edge"], u2=SET)  # rollers: free to contract in x
model.DisplacementBC(name="U1-fixed-center", createStepName="Initial", region=assembly.sets["Bottom-Center"], u1=SET)  # removes rigid-body x-translation only
model.DisplacementBC(
    name="Membrane-constraint",
    createStepName="Initial",
    region=assembly.sets["All-Nodes"],
    u3=SET, ur1=SET, ur2=SET,  # out-of-plane translation and the two in-plane-normal rotations
)
model.DisplacementBC(name="Applied-Tension", createStepName="Tension", region=assembly.sets["Top-Edge"], u2=TARGET_U2)  # the load

# ---------------------------------------------------------------------------------
# Output requests
# See abqpy_notes.md -- this Abaqus install's Model object has no FieldOutputRequest/
# HistoryOutputRequest Python API, so the auto-generated default keyword lines are edited
# directly via keywordBlock instead (a replace, not an insert -- inserting a second *Output
# block makes the Analysis Input File Processor abort with cascading undefined-node-set
# errors before it finishes resolving sets).
# ---------------------------------------------------------------------------------
model.keywordBlock.synchVersions(storeNodesAndElements=False)  # regenerate sieBlocks from the current model state
sie_blocks = model.keywordBlock.sieBlocks
field_output_index = next(i for i, line in enumerate(sie_blocks) if line.strip().startswith("*Output, field, variable=PRESELECT"))
model.keywordBlock.replace(
    field_output_index,
    "*Output, field\n"
    "*Element Output, ALLSECTIONPTS\n"  # ALLSECTIONPTS is essential for a composite shell: confirmed
    # live that Abaqus's plain default (this parameter omitted) writes output for only TWO of
    # this section's 24 section points -- the very top and very bottom of the whole laminate
    # stack -- silently discarding every interior ply's own data with no error or warning. The
    # section itself (`odb.sectionCategories`) always registers all 24 points correctly; it is
    # only the OUTPUT that was being truncated. Confirmed by directly probing a solved .odb:
    # without this parameter, `frame.fieldOutputs["S"].values` for one frame numbered exactly
    # 2 x n_elements entries (sectionPoint.number in {1, 24} only); with it, 24 x n_elements,
    # covering every ply. See abqpy_notes.md.
    "S, E, DAMAGEFT, DAMAGEFC, DAMAGEMT, DAMAGEMC, HSNFTCRT, HSNFCCRT, HSNMTCRT, HSNMCCRT\n"  # stress, strain, the four damage variables, the four Hashin criteria -- at every section point
    "*Node Output\n"
    "U, RF",
)
sie_blocks = model.keywordBlock.sieBlocks  # re-fetch: indices shift after the replace() above
history_output_index = next(i for i, line in enumerate(sie_blocks) if line.strip().startswith("*Output, history, variable=PRESELECT"))
model.keywordBlock.replace(
    history_output_index,
    "*Output, history\n"
    "*Node Output, nset=Top-Edge\n"
    "U2, RF2\n"  # the load-displacement history
    "*Energy Output\n"
    "ALLIE, ALLCD",  # internal energy and viscous-stabilization dissipation, to police the stabilization's cost (NB6 Part 5)
)

# ---------------------------------------------------------------------------------
# Job: submit and wait, timing the whole solve for NB6's cost discussion (Part 10)
# ---------------------------------------------------------------------------------
solve_start = time.time()
job = mdb.Job(name="OpenHoleTension", model="OpenHoleTension")
job.submit()
job.waitForCompletion()  # blocks until the solver process exits (success or failure -- see NB1 for why this doesn't raise on failure)
solve_seconds = time.time() - solve_start

# ---------------------------------------------------------------------------------
# Extract results to a plain JSON file (plus a companion damage CSV) for the notebook's
# live kernel to read afterward.
# ---------------------------------------------------------------------------------
from odbAccess import openOdb  # the API for reading a solved .odb, distinct from the "abaqus" module used to build the model

odb = openOdb("OpenHoleTension.odb")
step = odb.steps["Tension"]

# U2 is prescribed identically at every Top-Edge node, so any one of its per-node history
# regions gives the same displacement history; RF2 is NOT identical per node and must be
# summed across every "Node "-prefixed history region on Top-Edge (abqpy_notes.md: history
# output on a multi-node region is recorded per node, not summed, and this edge has tens of
# nodes, not the two NB1's single element had -- reading just one silently gives one node's
# share of the total reaction force).
top_edge_regions = [region for key, region in step.historyRegions.items() if key.startswith("Node ")]
u2_history = top_edge_regions[0].historyOutputs["U2"].data  # list of (time, value) pairs
rf2_per_node = [region.historyOutputs["RF2"].data for region in top_edge_regions]

# NOTE: `from abaqus import *` shadows the builtin `sum` with an Abaqus-provided symbol that
# rejects a generator argument -- accumulate manually instead. See abqpy_notes.md.
rf2_totals = []
for i in range(len(u2_history)):
    total = 0.0
    for node_data in rf2_per_node:
        total += node_data[i][1]
    rf2_totals.append(total)
u2 = [p[1] for p in u2_history]

# Energy history lives on the whole-model ('Assembly ASSEMBLY') history region, a single
# entry -- no per-node summing needed.
energy_regions = [region for key, region in step.historyRegions.items() if key.startswith("Assembly ")]
allie = [p[1] for p in energy_regions[0].historyOutputs["ALLIE"].data]
allcd = [p[1] for p in energy_regions[0].historyOutputs["ALLCD"].data]

# Peak load and the corresponding gross-section ultimate strength (Liu et al.: reaction
# force divided by the top-surface area, 76.2 mm x 1.0 mm laminate thickness -- the gross,
# not net, cross-section).
LAMINATE_THICKNESS = 8 * ply_thickness  # mm
peak_index = max(range(len(rf2_totals)), key=lambda i: rf2_totals[i])
f_peak_n = rf2_totals[peak_index]
u2_peak_mm = u2[peak_index]
sigma_ult_mpa = f_peak_n / (L * LAMINATE_THICKNESS)

# First-ply failure: the first frame at which any Hashin criterion reaches 1.0 anywhere in
# the model, read from the field-output frames (not the history output, which only carries
# U2/RF2/energy) -- mirrors NB1's own HSN*CRT >= 1 convention.
crit_names = ("HSNFTCRT", "HSNFCCRT", "HSNMTCRT", "HSNMCCRT")
mode_names = {"HSNFTCRT": "fiber tension", "HSNFCCRT": "fiber compression", "HSNMTCRT": "matrix tension", "HSNMCCRT": "matrix compression"}
n_plies = len(physical_angles)

fpf_frame_index = None
fpf_mode = None
fpf_ply = None
ply_damage_max = {"DAMAGEFT": [], "DAMAGEFC": [], "DAMAGEMT": [], "DAMAGEMC": []}  # per frame, per ply, max over all elements/section points in that ply

frames = step.frames
for frame_index, frame in enumerate(frames):
    for dmg_name in ply_damage_max:
        per_ply_max = [0.0] * n_plies
        for fv in frame.fieldOutputs[dmg_name].values:
            sp_number = fv.sectionPoint.number  # 1..(n_plies * 3), Abaqus's section-point numbering top to bottom
            ply_index = (sp_number - 1) // 3  # 0-based ply index; 3 section points per ply, per the section definition above
            if fv.data > per_ply_max[ply_index]:
                per_ply_max[ply_index] = fv.data
        ply_damage_max[dmg_name].append(per_ply_max)

    if fpf_frame_index is None:
        for crit_name in crit_names:
            for fv in frame.fieldOutputs[crit_name].values:
                if fv.data >= 1.0:
                    fpf_frame_index = frame_index
                    fpf_mode = mode_names[crit_name]
                    fpf_ply = (fv.sectionPoint.number - 1) // 3 + 1  # 1-based ply index
                    break
            if fpf_frame_index is not None:
                break

# ---------------------------------------------------------------------------------
# Companion damage CSV: per-element centroid + per-ply DAMAGEFT/DAMAGEMT, at two tagged
# frames (first-ply failure and the final frame), for the notebook's damage maps. Kept to
# two damage variables and two frames deliberately -- see the module docstring.
# ---------------------------------------------------------------------------------
out_dir = os.path.dirname(OUT_NAME)
if out_dir and not os.path.isdir(out_dir):  # OUT_NAME may be a bare filename -- nothing to create then.
    os.makedirs(out_dir)  # Abaqus's own Python kernel is Python 2, whose os.makedirs has no exist_ok kwarg

stem, _ = os.path.splitext(OUT_NAME)
damage_csv_path = stem + "_damage.csv"
tagged_frames = []
if fpf_frame_index is not None:
    tagged_frames.append(("fpf", fpf_frame_index))
tagged_frames.append(("final", len(frames) - 1))

element_centroids = {}  # element label -> (x, y), computed once from the undeformed node coordinates
for element in instance.elements:
    coords = [instance.nodes[node_label - 1].coordinates for node_label in element.connectivity]
    # NOTE: `from abaqus import *` shadows the builtin `sum` with a symbol that rejects a
    # generator argument (see abqpy_notes.md) -- accumulate manually instead.
    cx_total, cy_total = 0.0, 0.0
    for c in coords:
        cx_total += c[0]
        cy_total += c[1]
    element_centroids[element.label] = (cx_total / len(coords), cy_total / len(coords))

with open(damage_csv_path, "w") as f:
    f.write("frame_tag,element_label,x,y,ply,damage_ft,damage_mt\n")
    for tag, frame_index in tagged_frames:
        frame = frames[frame_index]
        ft_values = frame.fieldOutputs["DAMAGEFT"].values
        mt_values = {}
        for fv in frame.fieldOutputs["DAMAGEMT"].values:
            mt_values[(fv.elementLabel, fv.sectionPoint.number)] = fv.data
        for fv in ft_values:
            elem_label = fv.elementLabel
            sp_number = fv.sectionPoint.number
            ply_index = (sp_number - 1) // 3 + 1
            cx, cy = element_centroids.get(elem_label, (float("nan"), float("nan")))
            mt = mt_values.get((elem_label, sp_number), float("nan"))
            f.write("%s,%d,%.4f,%.4f,%d,%.6f,%.6f\n" % (tag, elem_label, cx, cy, ply_index, fv.data, mt))

# ---------------------------------------------------------------------------------
# Main results JSON -- self-describing: every CLI parameter echoed back, per the repo
# convention, plus everything NB6 needs for the load-displacement curve, the mesh-
# sensitivity study, and the physical-damage narrative.
# ---------------------------------------------------------------------------------
results = {
    "nhole": NHOLE,
    "n_elements": N_ELEMENTS,
    "n_circ": n_circ,
    "n_radial": n_radial,
    "nlgeom": "ON" if NLGEOM_ON else "OFF",
    "x_t": X_T, "x_c": X_C, "y_t": Y_T, "y_c": Y_C, "s_l": S_L, "s_t": S_T,
    "g_ft": G_FT, "g_fc": G_FC, "g_mt": G_MT, "g_mc": G_MC,
    "visc": VISC,
    "u2_target": TARGET_U2,
    "u2": u2,
    "rf2": rf2_totals,
    "allie": allie,
    "allcd": allcd,
    "f_peak_n": f_peak_n,
    "u2_peak_mm": u2_peak_mm,
    "sigma_ult_mpa": sigma_ult_mpa,
    "laminate_thickness_mm": LAMINATE_THICKNESS,
    "fpf": {
        "frame_index": fpf_frame_index,
        "u2_mm": u2[fpf_frame_index] if fpf_frame_index is not None else None,
        "rf2_n": rf2_totals[fpf_frame_index] if fpf_frame_index is not None else None,
        "sigma_mpa": (rf2_totals[fpf_frame_index] / (L * LAMINATE_THICKNESS)) if fpf_frame_index is not None else None,
        "mode": fpf_mode,
        "ply": fpf_ply,
    },
    "ply_damage_max": ply_damage_max,  # {damage_name: [[ply1..ply8] per frame]}
    "damage_csv": os.path.basename(damage_csv_path),
    "solve_seconds": solve_seconds,
}
with open(OUT_NAME, "w") as f:
    json.dump(results, f, indent=2)

odb.close()  # release the .odb file so the notebook (or a future run of this script) can read/overwrite it
