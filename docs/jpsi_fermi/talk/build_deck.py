from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import os

R = "/Volumes/wd_14tb/aao_norad_kin/jpsi_fermi/photo20M/report"
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "Jpsi_Fermi_photoproduction.pptx")

NAVY = RGBColor(0x1F, 0x2A, 0x44); WHITE = RGBColor(0xFF, 0xFF, 0xFF); ICE = RGBColor(0xC9, 0xD8, 0xEE)
INK = RGBColor(0x1A, 0x1A, 0x1A); GREY = RGBColor(0x5F, 0x63, 0x6B); TINT = RGBColor(0xEE, 0xF2, 0xF7)
BLUE = RGBColor(0x00, 0x72, 0xB2); ORANGE = RGBColor(0xD5, 0x55, 0x00); GREEN = RGBColor(0x00, 0x9E, 0x73); PINK = RGBColor(0xCC, 0x79, 0xA7)
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color


def box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(0.75)
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def text(slide, x, y, w, h, paras, size=13, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=4, margin=0.05, line_spacing=1.05):
    """paras: list of items; item = str or list of (text, dict) runs or dict(runs=..., bullet=bool, size=..)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin); tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for item in paras:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        p.alignment = align; p.space_after = Pt(space_after); p.line_spacing = line_spacing
        runs = item; psize = size; pcolor = color; pbold = bold; bullet = False; pfont = font
        if isinstance(item, dict):
            runs = item["runs"]; psize = item.get("size", size); pcolor = item.get("color", color)
            pbold = item.get("bold", bold); bullet = item.get("bullet", False); pfont = item.get("font", font)
        if isinstance(runs, str):
            runs = [(runs, {})]
        if bullet:
            runs = [("•  ", {"color": pcolor})] + list(runs)
        for t, opt in runs:
            r = p.add_run(); r.text = t
            r.font.name = opt.get("font", pfont); r.font.size = Pt(opt.get("size", psize))
            r.font.bold = opt.get("bold", pbold); r.font.italic = opt.get("italic", False)
            r.font.color.rgb = opt.get("color", pcolor)
            if opt.get("sup"):
                r.font._element.set("baseline", "30000")
            if opt.get("sub"):
                r.font._element.set("baseline", "-25000")
    return tb


def title(slide, t, sub=None, color=INK):
    text(slide, 0.5, 0.35, 12.3, 0.75, [t], size=30 if len(t) <= 52 else 26, font=HEAD, bold=True, color=color, anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        text(slide, 0.5, 1.0, 12.3, 0.4, [sub], size=14, color=GREY)


def image(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)


def footer(slide, n, dark=False):
    text(slide, 0.5, 7.08, 9, 0.3, ["V. Kubarovsky · Fermi motion in untagged J/ψ photoproduction · 27 September 2026"], size=9,
         color=ICE if dark else GREY)
    text(slide, 12.2, 7.08, 0.65, 0.3, [str(n)], size=9, color=ICE if dark else GREY, align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, heading, lines, size=12, head_color=BLUE, fill=TINT):
    box(slide, x, y, w, h, fill=fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    paras = [dict(runs=heading, size=size + 2, bold=True, color=head_color)] + [dict(runs=l, bullet=True) if not isinstance(l, dict) else l for l in lines]
    text(slide, x + 0.15, y + 0.1, w - 0.3, h - 0.2, paras, size=size, space_after=3)


def stat(slide, x, y, w, big, small, color=WHITE, small_color=ICE):
    text(slide, x, y, w, 0.9, [big], size=40, font=HEAD, bold=True, color=color, align=PP_ALIGN.LEFT)
    text(slide, x, y + 0.9, w, 0.9, [small], size=13, color=small_color)


def table(slide, x, y, w, rows, col_w=None, size=11, head_fill=NAVY, row_h=0.3):
    nr, nc = len(rows), len(rows[0])
    sh = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = sh.table
    tblPr = tbl._tbl.tblPr; tblPr.set("bandRow", "0"); tblPr.set("firstRow", "0")
    if col_w:
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            c.fill.solid(); c.fill.fore_color.rgb = head_fill if i == 0 else (WHITE if i % 2 else TINT)
            tf = c.text_frame; tf.paragraphs[0].text = ""; r = tf.paragraphs[0].add_run(); r.text = str(val)
            r.font.size = Pt(size); r.font.name = BODY; r.font.bold = (i == 0); r.font.color.rgb = WHITE if i == 0 else INK
            tf.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


# ------------------------------------------------------------------ 1 title
s = prs.slides.add_slide(BLANK); bg(s, NAVY)
text(s, 0.7, 0.9, 11.9, 1.6, ["Fermi motion in untagged J/ψ photoproduction on the deuteron"], size=40, font=HEAD, bold=True, color=WHITE)
text(s, 0.7, 2.55, 11.5, 0.9, ["Bremsstrahlung beam from the 5 cm target  ·  spectator model of the deuteron  ·  yield ratio to a free neutron at rest"], size=18, color=ICE)
text(s, 0.7, 3.4, 11.5, 0.5, ["V. Kubarovsky, Jefferson Lab   ·   27 September 2026   ·   generator exclurad_py, branch jpsi-fermi, module generator/photo.py"], size=13, color=ICE)
stat(s, 0.7, 4.6, 3.9, "0.994 ± 0.0003", "yield per neutron, deuteron / free neutron, E_γ > 8.6 GeV: Fermi motion changes it by less than 1 %")
stat(s, 4.9, 4.6, 3.6, "2.55 %", "of the J/ψ yield on the deuteron comes from photons below the free threshold 8.20 GeV")
stat(s, 8.8, 4.6, 3.9, "60 M events", "three 20 M-event samples (deuteron, free n, free p), 35 minutes on ten cores, zero majorant violations")

# ------------------------------------------------------------------ 2 generator
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "The generator: untagged photoproduction γ N → J/ψ N", "No scattered electron in the event. The beam radiates in the target, the photon strikes a nucleon, the final state is written to LUND.")
steps = [
    ("1", "Photon energy", "E_γ drawn from the bremsstrahlung spectrum, Tsai thin-target shape × Schiff endpoint factor, 6.5–10.6 GeV (tabulated inverse CDF)"),
    ("2", "Target nucleon", "at rest, or AV18 spectator momentum p_s: struck nucleon (M_D − E_s, −p_s) with off-shell mass m*; draw rejected if m* < 0.10 GeV"),
    ("3", "Accept", "s = (k + p_N)²; accept with σ_tot(s) · (1 − β_z) / w_max. σ_tot tabulated once; w_max is the strict head-on maximum, so no overweights"),
    ("4", "t and φ", "t from the exact inverse cumulative of dσ/dt at this s (401 points, physical limits), φ uniform"),
    ("5", "Kinematics", "J/ψ and recoil built in the γN centre of mass, boosted with the actual photon and nucleon four-vectors: k + p_D = J/ψ + N′ + p_s to 10⁻⁹ GeV"),
    ("6", "Output", "LUND: beam pid 22 with E_γ in the header, target 2112 / 1000010020, J/ψ → e⁺e⁻, recoil, spectator, vertex z along the cell"),
]
y0 = 1.55
for i, (n, h, d) in enumerate(steps):
    y = y0 + i * 0.9
    c = box(s, 0.5, y, 0.5, 0.5, fill=BLUE, shape=MSO_SHAPE.OVAL)
    text(s, 0.5, y, 0.5, 0.5, [n], size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0)
    text(s, 1.15, y - 0.06, 6.0, 0.85, [[(h + "  ", {"bold": True, "color": BLUE}), (d, {})]], size=12, space_after=0)
    if i < len(steps) - 1:
        box(s, 0.72, y + 0.52, 0.06, 0.36, fill=ICE)
card(s, 7.6, 1.55, 5.25, 1.75, "What is assumed on the hadron side", [
    "the electroproduction model at Q² = 0: in the Hand convention σ_T(Q²=0) is the real-photon dσ/dt, so nothing new is assumed",
    "neutron model jpsin.roberts2026 = proton model relabelled, σ_n = σ_p",
    "deuteron machinery of the fermi branch reused unchanged (generator/fermi.py)"], size=11)
card(s, 7.6, 3.45, 5.25, 1.75, "Validation", [
    "four-momentum conservation 5.5 × 10⁻⁹ GeV (the LUND precision), spectator on shell to 3 × 10⁻¹⁶",
    "free-proton E_γ spectrum vs n_γ(E_γ) σ_tot(W): χ²/ndf = 22.1/23",
    "t′ slope at 9.5 GeV: 1.64 vs the model’s 1.67 GeV⁻²",
    "speed 0.09 ms/event free nucleon, 0.65 ms/event deuteron (one core)"], size=11)
card(s, 7.6, 5.35, 5.25, 1.55, "Run", [
    dict(runs=[("python3 -m exclurad_py.generator.photo card.txt --jobs 10", {"font": "Courier New", "size": 10.5})]),
    "key = value card: model, ebeam, kmin/kmax, t range, nev, seed, fermi, fermi_mstar_min, target length and X₀, vz, maj_safety, outroot",
    "config.json per run: photons per electron, ⟨σ⟩, yield per electron per nucleon"], size=11)
footer(s, 2)

# ------------------------------------------------------------------ 3 photon beam
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Photon beam: thin-target bremsstrahlung and the endpoint")
image(s, R + "/fig_flux.png", 0.5, 1.15, w=12.3)
card(s, 0.5, 5.65, 3.95, 1.35, "Spectrum (Tsai, thin target)", [
    "dN/dE_γ = (X/X₀)(1/E_γ)[4/3(1−y)+y²]·S(E_γ), y = E_γ/E₀",
    "X = L/2: vertex density ∝ upstream thickness; LD₂ 3.25 × 10⁻³, LH₂ 2.81 × 10⁻³",
    "1.44 × 10⁻³ photons/e in 6.5–10.6 GeV, 0.77 × 10⁻³ above threshold"], size=10.5)
card(s, 4.65, 5.65, 4.35, 1.35, "Endpoint: complete screening is wrong there", [
    "q_min = m_e²E_γ/2EE′ outgrows 1/a_s as E′ → 0: screening switches off and Schiff’s bracket ln M + 1 − (2/b)arctan b falls through zero",
    "S(E_γ) = Schiff intermediate screening / complete-screening limit; screening radius = Bohr radius, 137 λ_C, for H and D"], size=10.5)
card(s, 9.2, 5.65, 3.65, 1.35, "Size of the correction", [
    "S = 0.97 / 0.91 / 0.77 / 0.58 / 0.36 at 300 / 100 / 30 / 10 / 3 MeV below E₀; zero at 0.42 MeV",
    "−1.24 % of the photons in 6.5–10.6 GeV; the last 100 MeV of the spectrum reshaped",
    "not included, each ≈ 1 MeV: dE/dx before radiating, beam spread, e–e endpoint"], size=10.5)
footer(s, 3)

# ------------------------------------------------------------------ 4 Fermi model
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Fermi motion: plane-wave spectator model with the AV18 deuteron")
image(s, D + "/av18.png", 0.7, 1.2, w=7.0)
table(s, 0.5, 4.6, 7.6, [
    ["E_γ [GeV]", "configurations above threshold", "⟨p_s⟩ [MeV/c]", "p_s > 300 MeV/c", "removed by floor 0.30", "floor 0.10"],
    ["7.0", "0.88 %", "373", "60.8 %", "2.9 %", "0.11 %"],
    ["7.5", "4.44 %", "224", "18.3 %", "0.7 %", "0.04 %"],
    ["8.0", "24.6 %", "117", "4.4 %", "0.17 %", "0.01 %"],
    ["8.2", "44.3 %", "91", "2.6 %", "0.10 %", "0.01 %"],
    ["10.0", "97.3 %", "86", "1.9 %", "0.06 %", "0.00 %"]], col_w=[0.9, 1.9, 1.2, 1.3, 1.3, 1.0], size=10, row_h=0.29)
text(s, 0.5, 6.4, 7.6, 0.6, ["Kinematics of the struck neutron for a photon along +z, AV18 unweighted (200 000 configurations): near threshold the high-momentum tail is the signal, so the m* floor was lowered from the π⁰/η value 0.30 GeV to 0.10 GeV."], size=10, color=GREY)
card(s, 8.5, 1.2, 4.35, 5.7, "Prescription", [
    "deuteron at rest; spectator on shell, p_s from AV18 (S + D), isotropic",
    "struck nucleon p_N = (M_D − E_s, −p_s), off shell: m*² = p_N² < M_n²",
    "s = m*² + 2E_γ(E_N − p_N,z): a neutron running toward the photon raises s, one running away lowers it",
    "cross section evaluated at the free-nucleon mass and the event’s invariants; m* enters only where the event is built",
    "the free threshold 8.20 GeV is no threshold: γd → J/ψ n p opens at 5.66 GeV; generated range starts at 6.5",
    "floor m* ≥ 0.10 GeV: removes 0.01 % of the draws, bounds the boost to the struck-nucleon frame",
    "not included: FSI, coherent production, off-shell modification of the amplitude, D-wave tensor structure, σ_n ≠ σ_p"], size=11.5)
footer(s, 4)

# ------------------------------------------------------------------ 5 cross section model
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Cross section model: roberts2026 against the CLAS12 data")
image(s, R + "/fig_model.png", 0.5, 1.2, w=8.3)
card(s, 9.1, 1.2, 3.75, 3.65, "Model", [
    "σ_tot = A (W − W_th)^α · exp[−B (W − W_th)]",
    "A = 1.018 nb, α = 0.707, B = −0.928 GeV⁻¹, W_th = M_N + M_J/ψ = 4.036 GeV",
    "dσ/dt = σ_tot · b · exp(b t′), b = 1.67 GeV⁻², t′ = t − t_min",
    "d ln σ/dW = 2.7 GeV⁻¹ at 10 GeV and diverges at threshold: shifts of s of 10⁻¹ GeV² are visible in the yield",
    "|t|_min = 1.8 / 0.62 / 0.36 GeV² at 8.5 / 9.5 / 10.5 GeV"], size=11)
card(s, 0.5, 5.05, 8.3, 1.85, "Against CLAS12 (Chatagnon et al. 2026, arXiv:2602.22128, Appendix A)", [
    "total cross section described within errors from threshold to 9.6 GeV; at 9.7–10.5 GeV the data lie 20–30 % above the curve, 1–2 σ per point",
    "t dependence: single exponential compatible with the data at |t| ≲ 2 GeV²; the points at 3–4 GeV² are a factor 2–3 above it",
    "the Fermi-motion ratio depends only on the slope of σ(W) near threshold, not on the normalisation; a refit of A, α, B, b is the next step for absolute yields"], size=11, head_color=ORANGE)
text(s, 9.1, 5.05, 3.75, 1.85, [dict(runs="Why this shape", bold=True, color=BLUE, size=13), "Near threshold σ ∝ (W − W_th)^0.7: convex and steep. That is what makes the deuteron yield sensitive to two small effects of opposite sign, binding and the flux-factor correlation (next slide)."], size=11)
footer(s, 5)

# ------------------------------------------------------------------ 6 result
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Result: J/ψ yield per neutron, deuteron over free neutron at rest")
image(s, R + "/fig_yield_ratio.png", 0.4, 1.15, w=8.5)
card(s, 9.1, 1.15, 3.75, 3.55, "Two effects of 4 %, opposite sign", [
    [("weight = n_γ(E_γ) · n(p_s) · (1 − β_z) · σ(s)", {"bold": True})],
    "binding: E_N = M_D − E_s = 0.933 < M_n, m* < M_n, so ⟨s⟩ at fixed E_γ is lower by 0.16 GeV² (W by 18 MeV): −4 to −5 %",
    "flux factor (1 − β_z): the Møller relative velocity of the moving neutron; averages to 1 but is correlated with s: +4 %",
    "net: between 0.99 and 1.00 at every E_γ above 8.6 GeV"], size=11)
table(s, 0.5, 4.95, 5.4, [
    ["E_γ [GeV]", "ratio", "without flux factor"],
    ["8.2–8.4", "1.776 ± 0.005", "1.62"], ["8.4–8.6", "1.088 ± 0.002", "1.015"], ["8.6–9.0", "0.999 ± 0.001", "0.949"],
    ["9.0–9.5", "0.990 ± 0.001", "0.952"], ["9.5–10.0", "0.993 ± 0.001", "0.961"], ["10.0–10.6", "0.996 ± 0.001", "0.968"]],
    col_w=[1.5, 2.0, 1.9], size=10, row_h=0.27)
card(s, 6.1, 4.95, 6.75, 1.95, "What it means", [
    "above 8.6 GeV a generator without Fermi motion is as good as one with it, to better than 1 % at every photon energy; mean 0.9944 ± 0.0003",
    "within 400 MeV of threshold the ratio is 1.09 and 1.78; below 8.20 GeV only the deuteron produces J/ψ: the threshold of σ(W) is smeared over ± 300 MeV and must be unfolded",
    "the first samples lacked the flux factor and gave 0.96: the cancellation is real and easy to miss"], size=11, head_color=GREEN)
footer(s, 6)

# ------------------------------------------------------------------ 7 below threshold
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "J/ψ below the free threshold")
image(s, R + "/fig_threshold.png", 0.4, 1.15, w=8.5)
card(s, 9.1, 1.15, 3.75, 3.55, "Numbers (20 M deuteron events)", [
    "2.55 % of the yield below 8.20 GeV, 1.41 % below 8.0, 0.33 % below 7.5",
    "1.02 × 10⁻⁵ nb × photons/e per neutron below threshold",
    "13 282 / 52 273 / 216 927 / 227 595 events in 6.5–7.0 / 7.0–7.5 / 7.5–8.0 / 8.0–8.2 GeV",
    "falls by an order of magnitude per 500 MeV: AV18 tail folded with the threshold rise of σ(W)"], size=11)
card(s, 0.5, 4.95, 5.4, 1.95, "Reliability", [
    "depends on exactly what is least certain: the AV18 tail beyond 300 MeV/c, the off-shell prescription, the absence of FSI",
    "read the sub-threshold yield as the plane-wave expectation, good to a factor of about two, not to percent"], size=11, head_color=ORANGE)
card(s, 6.1, 4.95, 6.75, 1.95, "Consequences", [
    "any threshold measurement on deuterium sees a smeared threshold; the generator provides the response to unfold it",
    "t below threshold: the events sit at the edge of phase space, |t|_min is large and the distribution is compressed toward it; the slope of a sub-threshold sample is not the elementary b",
    "the observable handle is the spectator proton (next slide)"], size=11, head_color=GREEN)
footer(s, 7)

# ------------------------------------------------------------------ 8 spectator
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "The spectator proton tags the sub-threshold events")
image(s, R + "/fig_spectator.png", 0.5, 1.15, w=12.3)
card(s, 0.5, 5.35, 3.95, 1.6, "Above threshold", [
    "⟨p_s⟩ = 99 MeV/c, 4.4 % above 300 MeV/c",
    "⟨m*⟩ = 0.915 GeV",
    "the bulk of AV18; the spectator is mostly undetectable"], size=11)
card(s, 4.65, 5.35, 4.35, 1.6, "Below threshold", [
    "all: ⟨p_s⟩ = 229 MeV/c, 26 % above 300 MeV/c, ⟨m*⟩ = 0.84 GeV",
    "below 8.0 GeV: 278 MeV/c, 36 %; below 7.5 GeV: 400 MeV/c, 68 %, ⟨m*⟩ = 0.71 GeV",
    "a fast proton at large angle: inside the CLAS12 acceptance"], size=11, head_color=ORANGE)
card(s, 9.2, 5.35, 3.65, 1.6, "In the sample", [
    "the spectator is written in every deuteron LUND event",
    "the floor m* ≥ 0.10 GeV removes 0.01 % of the draws, invisible in every distribution"], size=11, head_color=GREEN)
footer(s, 8)

# ------------------------------------------------------------------ 9 RC
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Do we need radiative corrections for photoproduction?")
image(s, D + "/flux_components.png", 0.5, 1.2, w=5.5)
text(s, 0.5, 5.15, 5.5, 1.8, ["Untagged 10.6 GeV electrons on a thin target: the quasi-real virtual photons of electroproduction with the electron undetected (equivalent-photon flux, Q² below 0.05–0.2 GeV²) outnumber the bremsstrahlung photons from the target by 8 to 11. The real CLAS12 sample is mostly quasi-real electroproduction."], size=11, color=GREY)
card(s, 6.3, 1.2, 6.55, 1.7, "1 · Real-photon process γN → J/ψ N: no", [
    "no lepton in the initial state, hence no initial-state radiation; the hadronic vertex corrections are absorbed in the measured σ",
    "what remains are flux corrections: endpoint (done), dE/dx before radiating, beam spread, thick-target multi-photon — each ≲ 1 % and only at the tip"], size=11)
card(s, 6.3, 3.05, 6.55, 1.95, "2 · J/ψ → e⁺e⁻(γ) final-state radiation: yes, in every mode", [
    "leading-log estimate: 21 % of the decays have m_ee < 3.0 GeV, 16 % below 2.9 GeV (μ⁺μ⁻: 7.5 % and 5.6 %)",
    "moves events out of a tight mass window and distorts the peak; GEMC does not radiate, so the generator must (PHOTOS-like decay radiation, channel independent)",
    "this is the radiative correction that matters for photoproduction"], size=11, head_color=ORANGE)
card(s, 6.3, 5.15, 6.55, 1.8, "3 · The actual CLAS12 configuration: electroproduction with RC", [
    "90 % of the untagged photons are virtual: use the electro path of exclurad_py (Q² → Q²_min, electron not required) with its radiative tail; the electron ISR/FSR changes the effective flux by a few %",
    "for the Fermi-motion ratio it is irrelevant: the flux cancels. photo.py remains the right tool for the bremsstrahlung part and for tagged real-photon beams"], size=11, head_color=GREEN)
footer(s, 9)

# ------------------------------------------------------------------ 10 summary
s = prs.slides.add_slide(BLANK); bg(s, NAVY)
text(s, 0.7, 0.5, 12, 0.8, ["Summary and next steps"], size=34, font=HEAD, bold=True, color=WHITE)
items = [
    [("New photoproduction mode of exclurad_py:", {"bold": True, "color": ICE}), (" bremsstrahlung beam with the correct endpoint, AV18 spectator deuteron, exact kinematics, strict majorant; 0.65 ms per event on the deuteron.", {})],
    [("The flux factor (1 − β_z) of the moving nucleon is essential:", {"bold": True, "color": ICE}), (" +4 %, and it cancels the −4 % binding effect. The same factor is missing in the electroproduction Fermi path (π⁰: 0.2 % on the yield, 1–3 % tilt in W_obs and y, 9 % in the tail below the box edge).", {})],
    [("Above 8.6 GeV Fermi motion changes the yield per neutron by less than 1 %", {"bold": True, "color": ICE}), (" (mean 0.9944 ± 0.0003, every 100 MeV bin within 0.99–1.00).", {})],
    [("Near and below threshold it is everything:", {"bold": True, "color": ICE}), (" 1.09 and 1.78 in the two bins above 8.20 GeV, 2.55 % of the yield below it, spectator protons of 230–400 MeV/c that CLAS12 can see.", {})],
    [("roberts2026 vs CLAS12:", {"bold": True, "color": ICE}), (" agrees to 9.6 GeV, 20–30 % low above; the t slope holds to 2 GeV². Refit A, α, B, b to the CLAS12 tables.", {})],
    [("Radiative corrections:", {"bold": True, "color": ICE}), (" none for the real-photon process itself; J/ψ → e⁺e⁻γ decay radiation (15–20 % below 3.0 GeV) must be added; for absolute CLAS12 yields the quasi-real electro path with RC is the right generator.", {})],
]
text(s, 0.7, 1.5, 12.0, 4.6, [dict(runs=r, bullet=True) for r in items], size=14, color=WHITE, space_after=9)
text(s, 0.7, 6.35, 12.0, 0.7, ["Report: docs/jpsi_fermi/report/jpsi_fermi_photoproduction.pdf (branch jpsi-fermi)  ·  OneDrive Work/2026_Jpsi_Fermi_photoproduction  ·  samples /Volumes/wd_14tb/aao_norad_kin/jpsi_fermi/photo20M"], size=11, color=ICE)
footer(s, 10, dark=True)

prs.save(OUT); print("wrote", OUT)
