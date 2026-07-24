"""
build_fiber_tension.py
-----------------------
Builds, solves, and extracts results for a single-element verification test of Abaqus's
built-in Hashin progressive damage model, loaded in pure fiber-direction tension.

See notebooks/01_fiber_tension_single_element.ipynb for the full theoretical background
and the citation for every parameter set below -- this script intentionally keeps only
short inline notes, not the full explanation. See notebooks/02_monte_carlo_fiber_strength.ipynb
for why X_T is treated as a random variable there, sweeping this same script across many
sampled values.

This script is independent of both notebooks and accepts its parameters on the command
line as `key=value` tokens (see "Parameters" below) -- it is never written to disk by a
notebook cell.

Do not import this module, and do not run it directly inside a live Jupyter kernel: the
`from abaqus import *` line below immediately hands the whole script off to a real Abaqus
installation and terminates the calling process (see CLAUDE.md for why). Run it either
directly (`python build_fiber_tension.py [key=value ...]`) or, from a notebook, via
`subprocess` so only the child process is affected.

Parameters (all optional, all `key=value` tokens, in any order):
    nlgeom=ON|OFF   geometric nonlinearity for the Static step (default ON)
    X_T=<float>     fiber tensile strength, MPa (default 990.0)
    U1=<float>      target displacement at the loaded edge, mm -- the applied
                    DisplacementBC's magnitude (default 3.0)
    out=<name>      output JSON filename, written to the current working directory
                    (default fiber_tension_results.json)

Example:
    python build_fiber_tension.py nlgeom=OFF X_T=990 U1=3.0 out=fiber_tension_results_off.json

Requires: pip install "abqpy==2023.*" (match the version to your installed Abaqus).
"""
import json
import sys

# ---------------------------------------------------------------------------------
# Parameter parsing -- BEFORE the abaqus import, so it runs identically on both of
# abqpy's self-relaunch passes (see the module docstring and CLAUDE.md).
#
# Deliberately NOT positional (sys.argv[1], sys.argv[-1], ...): on this install, real
# Abaqus's noGUI mode does not hand the script a plain [script, *args] argv the way
# ordinary Python does -- sys.argv here is the FULL underlying ABQcaeK.exe command line.
# Two things were confirmed live with a probe script (see abqpy_notes.md):
#   1. Anything passed after Abaqus's own "--" separator is appended at the very END,
#      not at a fixed index.
#   2. Abaqus's own launcher re-parses THOSE tokens too, not just its own options:
#      "nlgeom=OFF" (a name Abaqus itself recognizes as a keyword) survives as one
#      "key=value" token, but "X_T=990" (a name it does NOT recognize) gets rewritten
#      into TWO separate tokens, '-X_T' and '990' -- i.e. any unrecognized "name=value"
#      becomes "-name" followed by "value" as its own list entry. Confirmed live:
#      ['...ABQcaeK.exe', '-cae', '-noGUI', 'script.py', '-lmlog', 'ON', '-tmpdir',
#      '...', 'nlgeom=OFF', '-X_T', '990', '-U1', '3.0', '-out', 'results.json'].
# The scan below handles both shapes: a single "key=value" token (the first, plain-
# Python pass, and any Abaqus-recognized keyword on the second pass), or a "-key" token
# immediately followed by a separate "value" token (any unrecognized keyword on the
# second pass). None of Abaqus's own injected option names collide with ours, so
# unrelated tokens (-lmlog, -tmpdir, ...) are silently skipped either way.
# ---------------------------------------------------------------------------------
PARAMS = {"nlgeom": "ON", "x_t": "990.0", "u1": "3.0", "out": "fiber_tension_results.json"}
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

NLGEOM_ON = PARAMS["nlgeom"].upper() != "OFF"  # anything but a literal "OFF" is treated as ON
X_T = float(PARAMS["x_t"])  # MPa
TARGET_U1 = float(PARAMS["u1"])  # mm
OUT_NAME = PARAMS["out"]

from abaqus import *  # triggers the self-relaunch under real Abaqus -- see the module docstring
from abaqusConstants import *  # symbolic constants used throughout (ON, OFF, LAMINA, S4R, ...)
import mesh  # mesh.ElemType, used below to force the S4R element choice

# ---------------------------------------------------------------------------------
# Geometry: a single 10 mm x 10 mm square shell "ply"
# ---------------------------------------------------------------------------------
mdb.Model(name="FiberTension")  # create a new, empty model database entry named "FiberTension"
model = mdb.models["FiberTension"]  # keep a handle to it for everything that follows

sketch = model.ConstrainedSketch(name="ply_profile", sheetSize=20.0)  # a 2D sketch, canvas 20 mm across
sketch.rectangle(point1=(0.0, 0.0), point2=(10.0, 10.0))  # draw the 10x10 mm square profile

part = model.Part(name="Ply", dimensionality=THREE_D, type=DEFORMABLE_BODY)  # an empty deformable part
part.BaseShell(sketch=sketch)  # extrude the sketch into a shell (no thickness yet -- that comes from the section below)

# ---------------------------------------------------------------------------------
# Material: E-glass/epoxy unidirectional lamina (Nelson, Riddle & Cairns, 2017)
# ---------------------------------------------------------------------------------
material = model.Material(name="GlassEpoxy_UD")  # create the material definition; every property below attaches to it

material.Elastic(
    table=((40600.0, 16300.0, 0.27, 16800.0, 16800.0, 6000.0),),  # E1, E2, nu12, G12, G13, G23 [MPa]
    type=LAMINA,  # orthotropic engineering-constants form (as opposed to, e.g., isotropic or full anisotropic)
)

material.HashinDamageInitiation(
    table=((X_T, 582.0, 60.0, 35.0, 112.0, 124.0),),  # XT, XC, YT, YC, SL, ST [MPa] -- XT is the CLI parameter above; the rest are the notebook's fixed values
    alpha=1.0,  # shear-influence coefficient in the fiber-tension criterion -- alpha=1 recovers Hashin (1980);
    # NOTE: abqpy's local type stub for this method also lists several MSFLD-criterion
    # parameters (feq, fnn, frequency, ...) that do not apply to Hashin -- a known
    # documentation-generation bug. Only `table` and `alpha` are real Hashin parameters.
    # See abqpy_notes.md.
)

material.hashinDamageInitiation.DamageEvolution(
    type=ENERGY,  # the evolution law is driven by fracture energy, not by a displacement value directly
    table=((1290.0, 757.0, 78.0, 45.5),),  # GfT, GfC, GmT, GmC [N/mm] -- literal source values, see NB1
    softening=LINEAR,  # stress drops linearly with equivalent displacement once damage initiates -- see NB1
)

# ---------------------------------------------------------------------------------
# Section + material orientation
# ---------------------------------------------------------------------------------
model.HomogeneousShellSection(name="Ply-Section", material="GlassEpoxy_UD", thickness=0.2)  # gives the shell its 0.2 mm thickness and attaches the material to it
part.SectionAssignment(region=(part.faces,), sectionName="Ply-Section")  # apply that section to the whole part (its one face)
part.MaterialOrientation(region=(part.faces,), orientationType=GLOBAL, axis=AXIS_3)  # material axis 1 (fibers) := global X, material axis 2 := global Y

# ---------------------------------------------------------------------------------
# Mesh the part, THEN instance it: force exactly one S4R element
# See abqpy_notes.md -- on this install, a dependent instance created before its part
# is meshed does not pick up the mesh afterward, so the part must be meshed first.
# ---------------------------------------------------------------------------------
part.seedPart(size=20.0)  # larger than the part -> exactly one element per edge
part.setElementType(
    regions=(part.faces,),
    elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),),  # 4-node reduced-integration shell -- see NB1 for what "reduced integration" means
)
part.generateMesh()  # actually mesh the part with the seed size and element type set above

assembly = model.rootAssembly  # the (initially empty) assembly that instances live in
instance = assembly.Instance(name="Ply-1", part=part, dependent=ON)  # place one copy of the meshed part into the assembly; "dependent" means it shares the part's mesh rather than copying it

# ---------------------------------------------------------------------------------
# Node sets, defined by position (robust to whatever node numbering the mesher assigns)
# ---------------------------------------------------------------------------------
tol = 1e-3  # mm; tolerance for the bounding-box node searches below
big = 1e6  # mm; a bound far outside the model, used to make a bounding box effectively unbounded in one direction
nodes = instance.nodes
x0_nodes = nodes.getByBoundingBox(xMin=-tol, xMax=tol, yMin=-big, yMax=big, zMin=-big, zMax=big)  # the two nodes on the x=0 edge (fixed edge)
x10_nodes = nodes.getByBoundingBox(xMin=10 - tol, xMax=10 + tol, yMin=-big, yMax=big, zMin=-big, zMax=big)  # the two nodes on the x=10 edge (loaded edge)
y0_nodes = nodes.getByBoundingBox(xMin=-big, xMax=big, yMin=-tol, yMax=tol, zMin=-big, zMax=big)  # the two nodes on the y=0 edge

assembly.Set(name="X0-Edge", nodes=x0_nodes)  # register each bounding-box result as a named Abaqus Set, so it can be referenced by name in BCs/step/output requests below
assembly.Set(name="X10-Edge", nodes=x10_nodes)
assembly.Set(name="Y0-Edge", nodes=y0_nodes)

# ---------------------------------------------------------------------------------
# Step: Static, General (displacement-controlled) -- see NB1 for why.
# TARGET_U1 and NLGEOM_ON are the CLI parameters parsed above. With the default Ramp
# amplitude, step time 0->1 maps linearly onto edge displacement 0->TARGET_U1 (applied
# below), so automatic incrementation in step time is equivalently automatic incrementation
# in displacement.
# ---------------------------------------------------------------------------------
model.StaticStep(
    name="Tension",
    previous="Initial",  # this step follows Abaqus's implicit "Initial" step (where the BCs below that use createStepName="Initial" are defined)
    nlgeom=ON if NLGEOM_ON else OFF,  # geometric nonlinearity -- see NB1 for what changes when this is OFF vs ON
    timePeriod=1.0,  # the step runs from time 0 to 1 -- arbitrary in itself, meaningful only through the Ramp amplitude mapping it onto displacement below
    initialInc=0.01,  # increment size to attempt first
    minInc=1e-5,  # smallest increment Abaqus is allowed to shrink to before giving up
    maxNumInc=100,  # upper bound on how many increments the step is allowed to take
)

# ---------------------------------------------------------------------------------
# Boundary conditions -- an essential (rigid-body-removing) scheme, consistent with the
# target uniform field U1=k1*x, U2=k2*y (see NB1)
# ---------------------------------------------------------------------------------
model.DisplacementBC(name="U1-fixed", createStepName="Initial", region=assembly.sets["X0-Edge"], u1=SET)  # U1=0 on the fixed edge -- removes x-translation and in-plane (about Z) rotation
model.DisplacementBC(name="U2-fixed", createStepName="Initial", region=assembly.sets["Y0-Edge"], u2=SET)  # U2=0 on the y=0 edge -- removes y-translation, consistent with U2=k2*y=0 there
model.DisplacementBC(name="U3-fixed-X0", createStepName="Initial", region=assembly.sets["X0-Edge"], u3=SET)  # U3=0 on the x=0 edge ...
model.DisplacementBC(name="U3-fixed-Y0", createStepName="Initial", region=assembly.sets["Y0-Edge"], u3=SET)  # ... and on the y=0 edge -- together, three non-collinear points remove the two out-of-plane rotations and z-translation, without constraining U3 anywhere else
model.DisplacementBC(name="Applied-Tension", createStepName="Tension", region=assembly.sets["X10-Edge"], u1=TARGET_U1)  # the load: ramp U1 at the x=10 edge up to TARGET_U1 over the "Tension" step

# ---------------------------------------------------------------------------------
# Output requests
# See abqpy_notes.md -- this Abaqus install's Model object has no FieldOutputRequest/
# HistoryOutputRequest Python API, so the auto-generated default keyword lines are
# edited directly via keywordBlock instead (a replace, not an insert -- see the notes).
# ---------------------------------------------------------------------------------
model.keywordBlock.synchVersions(storeNodesAndElements=False)  # regenerate model.keywordBlock.sieBlocks from the current model state
sie_blocks = model.keywordBlock.sieBlocks  # the full list of raw *KEYWORD lines Abaqus will write to the .inp file
field_output_index = next(i for i, line in enumerate(sie_blocks) if line.strip().startswith("*Output, field, variable=PRESELECT"))  # find the auto-generated default field-output line
model.keywordBlock.replace(
    field_output_index,  # ... and replace that one line/block ...
    "*Output, field\n"
    "*Element Output\n"
    "S, E, DAMAGEFT, DAMAGEFC, DAMAGEMT, DAMAGEMC, HSNFTCRT, HSNFCCRT, HSNMTCRT, HSNMCCRT\n"  # stress, strain, the four damage variables, the four Hashin criteria
    "*Node Output\n"
    "U, RF",  # nodal displacements and reaction forces
)
sie_blocks = model.keywordBlock.sieBlocks  # re-fetch: indices shift after the replace() above
history_output_index = next(i for i, line in enumerate(sie_blocks) if line.strip().startswith("*Output, history, variable=PRESELECT"))  # find the auto-generated default history-output line
model.keywordBlock.replace(
    history_output_index,
    "*Output, history\n"
    "*Node Output, nset=X10-Edge\n"
    "U1, U2, RF1",  # the three quantities the notebook actually reads back afterward
)

# ---------------------------------------------------------------------------------
# Job: submit and wait
# ---------------------------------------------------------------------------------
job = mdb.Job(name="FiberTension", model="FiberTension")  # define a job that solves the "FiberTension" model
job.submit()  # hand the generated input file to the Abaqus/Standard solver
job.waitForCompletion()  # block until the solver process exits (success or failure -- see NB1 for why this doesn't raise on failure)

# ---------------------------------------------------------------------------------
# Extract results to a plain JSON file for the notebook's live kernel to read afterward.
# See abqpy_notes.md -- the "X10-Edge" node set has TWO nodes (the two corners of the
# loaded edge), and Abaqus records history output per node, not per set, so RF1 has to
# be summed across both nodes to get the total edge reaction force.
# ---------------------------------------------------------------------------------
from odbAccess import openOdb  # the API for reading a solved .odb results database (distinct from the "abaqus" module used to build the model)

odb = openOdb("FiberTension.odb")  # open the results database the job above just wrote
step = odb.steps["Tension"]  # the one analysis step we ran; odb.steps also has an "Initial" entry with no results

# step.historyRegions is a dict keyed by region name (one entry per node/region a history
# output was requested on); pick out just the two per-node regions on the loaded edge,
# excluding the 'Assembly ASSEMBLY' region (which isn't needed here).
node_regions = [region for key, region in step.historyRegions.items() if key.startswith("Node ")]
u1_history = node_regions[0].historyOutputs["U1"].data  # a list of (time, value) pairs, one per increment; U1 is prescribed identically at both nodes, so either region gives the same values
rf1_per_node = [region.historyOutputs["RF1"].data for region in node_regions]  # the two nodes' individual reaction-force histories, to be summed below

# NOTE: `from abaqus import *` shadows the builtin `sum` with an Abaqus-provided symbol
# of the same name that rejects a generator argument -- accumulate manually instead.
# See abqpy_notes.md.
rf1_totals = []
for i in range(len(u1_history)):  # for each increment ...
    total = 0.0
    for node_data in rf1_per_node:  # ... add up that increment's RF1 across both loaded-edge nodes ...
        total += node_data[i][1]  # node_data[i] is a (time, value) pair; [1] takes the value
    rf1_totals.append(total)  # ... giving the total reaction force on the whole edge

# step.frames holds one entry per increment Abaqus saved a field-output snapshot for;
# frame.fieldOutputs[<name>].values[0].data reads the single value for our one element
# (there is only one integration point, since the whole model is one S4R element -- a
# model with more elements would need to loop over .values instead of indexing [0]).
damage_ft, damage_mt, hsn_ft, hsn_mt = [], [], [], []
for frame in step.frames:
    damage_ft.append(frame.fieldOutputs["DAMAGEFT"].values[0].data)  # fiber-tension damage variable d_ft
    damage_mt.append(frame.fieldOutputs["DAMAGEMT"].values[0].data)  # matrix-tension damage variable d_mt
    hsn_ft.append(frame.fieldOutputs["HSNFTCRT"].values[0].data)  # fiber-tension Hashin criterion F_ft
    hsn_mt.append(frame.fieldOutputs["HSNMTCRT"].values[0].data)  # matrix-tension Hashin criterion F_mt

results = {
    "nlgeom": "ON" if NLGEOM_ON else "OFF",  # the CLI parameters this run actually used -- carried in the JSON
    "x_t": X_T,  # itself, so every run file is self-describing, not just named by convention
    "u1_target": TARGET_U1,
    "u1": [p[1] for p in u1_history],  # drop the time component of each (time, value) pair -- keep just the displacement values
    "rf1": rf1_totals,
    "damage_ft": damage_ft,
    "damage_mt": damage_mt,
    "hsn_ft": hsn_ft,
    "hsn_mt": hsn_mt,
}
with open(OUT_NAME, "w") as f:  # write to the current working directory (scripts/, when launched via a notebook's subprocess call)
    json.dump(results, f, indent=2)

odb.close()  # release the .odb file so the notebook (or a future run of this script) can read/overwrite it
