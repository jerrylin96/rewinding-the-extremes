// Build from this directory with: typst compile poster.typ output/pdf/run-it-back.pdf
// The plots use manuscript panels: ageostrophic_compare_L500 (a–c), the
// top row of each storm track figure, and the heatwave time series (q, r).
#set document(title: "Run it Back: Autoregression vs. Video Diffusion for Final-Conditioned Weather Trajectories", author: ("Jerry Lin", "Elizabeth A. Barnes"))
#set page(width: 61cm, height: 85cm, margin: 0mm)
#let navy = rgb("#143446")
#let ink = rgb("#182F3A")
#let teal = rgb("#007C72")
#let mint = rgb("#91E2CF")
#let rust = rgb("#A44422")
#let pale = rgb("#EFF5F3")
#let rule = rgb("#B6C9C8")
#let muted = rgb("#405A63")
#set text(font: ("Avenir Next", "Arial"), size: 23pt, fill: ink, lang: "en")
#set par(leading: 0.45em, spacing: 0.6em)
#let t(size, body, color: ink, weight: "regular") = text(size: size, fill: color, weight: weight, body)
#let tag(body, color: teal) = text(size: 18pt, tracking: 1.2pt, weight: "bold", fill: color, body)
#let head(body) = t(37pt, body, weight: "bold")
#let small(body) = t(18pt, body, color: muted)
#let at(x, y, w, body) = place(top + left, dx: x, dy: y, block(width: w, body))
#let divider(y) = at(18mm, y, 574mm, line(length: 100%, stroke: 1.4pt + rule))

// The paths in the header are a schematic of conditioning, not sampled tracks.
#let histories = box(width: 148mm, height: 56mm)[
  #for path in (
    ((0mm, 1mm), (26mm, 6mm), (49mm, 4mm), (71mm, 12mm), (99mm, 14mm), (122mm, 23mm), (144mm, 28mm)),
    ((0mm, 11mm), (26mm, 10mm), (49mm, 15mm), (71mm, 13mm), (99mm, 21mm), (122mm, 24mm), (144mm, 28mm)),
    ((0mm, 22mm), (26mm, 19mm), (49mm, 24mm), (71mm, 20mm), (99mm, 22mm), (122mm, 25mm), (144mm, 28mm)),
    ((0mm, 32mm), (26mm, 27mm), (49mm, 31mm), (71mm, 31mm), (99mm, 27mm), (122mm, 28mm), (144mm, 28mm)),
    ((0mm, 44mm), (26mm, 38mm), (49mm, 42mm), (71mm, 34mm), (99mm, 32mm), (122mm, 30mm), (144mm, 28mm)),
    ((0mm, 55mm), (26mm, 51mm), (49mm, 43mm), (71mm, 44mm), (99mm, 36mm), (122mm, 32mm), (144mm, 28mm)),
  ) {
    for j in range(path.len() - 1) {
      place(top + left, line(start: path.at(j), end: path.at(j+1), stroke: 1.9pt + mint))
    }
  }
  #place(top + left, dx: 141mm, dy: 25mm, circle(radius: 3mm, fill: white))
]

#at(0mm, 0mm, 610mm, rect(width: 100%, height: 108mm, fill: navy, stroke: none))
#at(18mm, 10mm, 574mm, tag([TACKLING CLIMATE CHANGE WITH MACHINE LEARNING   /   NEURIPS 2026], color: mint))
#at(17mm, 24mm, 400mm, t(79pt, [Run it Back], color: white, weight: "bold"))
#at(19mm, 56mm, 420mm, t(34pt, [Autoregression vs. Video Diffusion for\ Final-Conditioned Weather Trajectories], color: white, weight: "medium"))
#at(19mm, 84mm, 412mm, t(23pt, [*Jerry Lin*  and  *Elizabeth A. Barnes*     ·     Boston University], color: white))
#at(19mm, 96mm, 412mm, t(19pt, [Presented by Zeyuan Hu, NVIDIA], color: mint))
#at(443mm, 23mm, 148mm, histories)
#at(443mm, 85mm, 153mm, t(17pt, [−66 hours], color: mint) + h(1fr) + t(17pt, [Pinned event], color: white))

#at(18mm, 118mm, 574mm, head([Which weather patterns could lead to the same extreme?]))
#at(18mm, 139mm, 574mm, t(23pt, [Each model samples 1,000 possible 66-hour histories per event, conditioned on the final weather state from ERA5 reanalysis.]))
#at(18mm, 162mm, 277mm, [
  #tag([REWIND-AR])
  #v(1mm)
  #t(21pt, [A probabilistic FourCastNet 3 model starts at the exact ERA5 state and steps back six hours at a time on a roughly 25 km grid.])
])
#at(315mm, 162mm, 277mm, [
  #tag([CBOTTLE-VIDEO], color: rust)
  #v(1mm)
  #t(21pt, [A video diffusion model samples all 12 frames together on a roughly 100 km grid. It reproduces the endpoint approximately.])
])
#divider(205mm)

#at(18mm, 214mm, 574mm, tag([PHYSICAL CONSISTENCY   /   THREE EVENTS]))
#at(18mm, 227mm, 574mm, head([Rewind-AR stays close to ERA5’s midlatitude wind balance]))
#at(18mm, 248mm, 574mm, t(23pt, [At 500 hPa, we subtract the geostrophic wind inferred from height gradients from the wind at each grid point. The ageostrophic fraction is the ratio of mean residual speed to mean wind speed over 30–60° in both hemispheres.]))
#let panel-title(x, width, body) = at(x, 281mm, width, align(center, t(22pt, body, weight: "semibold")))
#panel-title(47mm, 171mm, [Hurricane Ian])
#panel-title(231mm, 170mm, [Pacific Northwest heatwave])
#panel-title(415mm, 170mm, [Superstorm Sandy])
#at(18mm, 294mm, 574mm, image("figures/ageostrophic-fraction.png", width: 100%, alt: "Three time series of 500 hPa ageostrophic wind fraction. The Rewind-AR ensemble mean follows native ERA5, while cBottle-video lies above its grid-matched ERA5 reference in each event."))
// The first and last data points are inset from the plot borders.
#for (start, end) in ((54mm, 210mm), (240mm, 395mm), (425mm, 581mm)) {
  at(start - 9mm, 388mm, 27mm, small([−66 h]))
  at(end - 25mm, 388mm, 31mm, align(right, small([0 h · event])))
}
#at(18mm, 408mm, 575mm, t(22pt, [Thin lines show members and bold lines show ensemble means. Averaged over the 11 frames before each event, Rewind-AR’s fraction is *0.002–0.005 below* native ERA5. cBottle-video’s is *0.06–0.09 above* its grid-matched ERA5 reference. Rewind-AR’s winds still average 1.1–1.8 m/s too weak at −66 h.]))
#divider(446mm)

#let storm(x, case, headline, asset, members, description, alt) = {
  let w = 276mm
  at(x, 455mm, w, tag(case))
  at(x, 468mm, w, t(30pt, headline, weight: "bold"))
  at(x, 501mm, w, grid(columns: (1fr, 1fr), align: center, ..members.map(it => t(18pt, it, weight: "semibold"))))
  at(x, 514mm, w, image(asset, width: w, alt: alt))
  at(x, 626mm, w, t(21pt, description))
}
#storm(18mm, [SUPERSTORM SANDY   /   OCTOBER 2012],
  [Some Rewind-AR storms start implausibly far inland],
  "figures/sandy-tracks.png", ([Rewind-AR · 974 tracks], [cBottle-video · 788 tracks]),
  [All 1,000 Rewind-AR members pass the stability screen. Among the 974 retained tracks, some start well inland over the southeastern US before reaching New Jersey. Sandy falls within both models’ training periods.],
  "On the left, some purple Rewind-AR tracks begin inland in the southeastern United States. The ERA5 track begins offshore.")
#storm(315mm, [HURRICANE IAN   /   SEPTEMBER 2022],
  [Height patterns explain Ian’s starting longitude],
  "figures/ian-tracks.png", ([Rewind-AR · 867 tracks], [cBottle-video · 947 tracks]),
  [At −66 h, eight principal components of the 500 hPa height field explain *87%* of the variance in Ian’s longitude for Rewind-AR and *6%* for cBottle-video (10-fold cross-validation). The height field can also reflect the storm’s own position.],
  "Hurricane Ian tracks converge at its South Carolina landfall. Rewind-AR starting locations span both sides of Florida.")
// Mark one visible inland endpoint in the original Sandy track panel.
#at(37mm, 540mm, 60mm, t(15pt, [Inland start], color: rust, weight: "bold"))
#at(47mm, 547mm, 16mm, line(end: (14.6mm, 11.1mm), stroke: 1.3pt + rust))
#at(60.5mm, 556.9mm, 3mm, circle(radius: 1.2mm, stroke: 1.4pt + rust, fill: none))
#at(18mm, 666mm, 574mm, small([Track colors rank members by the first height component, from purple to yellow for Rewind-AR and cyan to magenta for cBottle-video. Black traces ERA5 and gold the closest member.]))
#divider(685mm)

#at(18mm, 693mm, 574mm, tag([PACIFIC NORTHWEST HEATWAVE   /   JUNE 2021]))
#at(18mm, 705mm, 574mm, t(30pt, [Rewind-AR captures the rise in daily peak temperature], weight: "bold"))
// Each cropped plot retains its source y scale. Relabel in print-size type.
#let heat(x, name, label, pixels-high, ticks) = {
  let w = 250mm
  let y = 736mm
  at(x, 725mm, w, t(17pt, label, weight: "semibold") + h(1fr) + small([Land-mean 2 m temperature (K)]))
  at(x, y, w, image("figures/heatwave-" + name + ".png", width: w, alt: label))
  for (value, row) in ticks {
    at(x - 15mm, y + w * row / 1862 - 2.4mm, 13mm, align(right, t(15pt, str(value), color: muted)))
  }
  for (h, label) in ((0, [Jun 25]), (24, [Jun 26]), (48, [Jun 27]), (66, [Jun 28])) {
    at(x + w * h / 66 - 12mm, y + w * pixels-high / 1862 + 2mm, 24mm, align(center, t(15pt, label, color: muted)))
  }
}
#heat(34mm, "rewind", "Rewind-AR · 778 members", 263, ((305, 19.5), (300, 79.5), (295, 139.5), (290, 199.5), (285, 259.5)))
#heat(329mm, "cbottle", "cBottle-video · 1,000 members", 262, ((305, 17.5), (300, 84.5), (295, 151.5), (290, 218.5)))
#at(18mm, 785mm, 574mm, t(18pt, [Rewind-AR starts 1.9 K cooler than ERA5 on average. Its daily peaks rise, but daytime temperatures average 1.0–1.6 K too cool on June 26–27. All cBottle-video members start warmer than ERA5. Their daily peaks stay nearly level. Dashed black marks ERA5. Blue and red highlight the coolest and warmest starting members. Dates are local (PDT).]))
#at(0mm, 815mm, 610mm, rect(width: 100%, height: 35mm, fill: pale, stroke: none))
#at(18mm, 820mm, 574mm, small([Stability screening removed 22.2% of Rewind-AR heatwave members and 13.3% for Ian. Model resolution and training data also differ.]))
#at(18mm, 836mm, 574mm, grid(columns: (auto, 1fr, auto), column-gutter: 12mm,
  small(link("https://github.com/jerrylin96/run-it-back")[*Training*  github.com/jerrylin96/run-it-back]),
  small(link("https://github.com/jerrylin96/rewinding-the-extremes")[*Analysis*  github.com/jerrylin96/rewinding-the-extremes]),
  small(link("mailto:jlin404@bu.edu")[#text("jlin404@bu.edu")])
))
