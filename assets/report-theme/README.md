# Shared project-report presentation

The report theme follows the Fraunces / IBM Plex typography and cream-and-forest editorial palette used by jeremycleland.com and clelandco.com. Fonts are self-hosted; SIL Open Font licenses are retained in `fonts/licenses/`.

`report-theme.css` supplies light/dark tokens, report typography, page navigation, responsive tables, focus states and print styles. `report-theme.js` applies the system color preference or saved choice, adds mobile contents navigation, and labels responsive table cells. Static figures and recorded results retain their original values and appearance. The three public report repositories carry identical copies of these shared assets so publication does not depend on another project's runtime.

Theme choice persists across project reports on the same GitHub Pages origin. Browsers without JavaScript receive the light reading theme and the original report content. Figures remain available at their existing source URLs.

Legacy inline styles retain page-specific geometry. The shared stylesheet loads last and supplies the published palette, typography and presentation. When changing the shared theme, synchronize the three copies and verify representative report families at mobile, tablet and desktop sizes in both color themes.

The PlantDoc training template references assets from the standard `outputs/<experiment>/reports/` directory. A report rendered to a custom directory needs its relative theme links adjusted or the theme assets bundled alongside it.
