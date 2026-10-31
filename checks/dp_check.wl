(* Independent exact re-solve of the end-game game on a short horizon, written from the model
   specification in src/nbaeg/dp.py (docstring), not from its loops: memoized recursion over exact
   rationals. Compared state by state with the numba solver (checks/dp_small_python.json).
   Part 2: symbolic last-FT break-even identity, checked on the full fitted model.
   usage: wolframscript -file checks/dp_check.wl   (run from the repo root) *)
$RecursionLimit = 100000;
dir = DirectoryName[$InputFileName];
in = Import[FileNameJoin[{dir, "dp_small_inputs.json"}], "RawJSON"];
r[x_] := x/1000;
{M2, M3, X2, X3, NONE} = r /@ (in /@ {"M2", "M3", "X2", "X3", "NONE"});
ALLOW2 = in["ALLOW2"]; FE = in["FE"]; CTO = in["C_TO"]; RT = in["RT"]; SC = in["SC"];
(* rebound-delay distributions of ordinary and deliberate FT misses: delays (steps) and weights *)
RTO = in["RTO"]; RTD = in["RTD"]; WO = r /@ in["WO"]; WD = r /@ in["WD"];
{psf, ptov, rfg, thfg, etafg, rft, rdm, thft, thdm, etaft, etadm, q, psf3, ppre, ot} =
  r /@ (in /@ {"p_sf", "p_tov", "rho_fg", "th_fg", "eta_fg", "rho_ft", "rho_dm", "th_ft", "th_dm", "eta_ft", "eta_dm",
     "q", "p_sf3", "p_pre", "ot"});
DM = in["DM"]; NS = in["NS"];
SCT = Max[1, SC - CTO];   (* shot clock left after a live timeout that follows an offensive rebound *)
(* remaining shot clock by start type: OREB and FC14 (side-out, reset to max(left, 14 s)) SC, FC14T (timeout,
   clock keeps running) SC - CTO, otherwise none (0) *)
clk[s_] := Which[s == 3 || s == 4, SC, s == 5, SCT, True, 0];
cl[d_] := Clip[d, {-DM, DM}];
term[d_] := Which[d > 0, 1, d < 0, 0, True, ot];

(* value of the team with the ball, at a possession phase *)
v[n_, d_, s_, a_, b_, ga_, gb_] /; d =!= cl[d] := v[n, cl[d], s, a, b, ga, gb];
(* with probability ppre the offence beats a committed foul: it may release its best shot after
   tau < FE[s] steps (early[...]) instead of accepting the foul *)
v[n_, d_, s_, a_, b_, ga_, gb_] := v[n, d, s, a, b, ga, gb] = Module[{val, nf, vf, p3},
   If[n <= 0, Return[term[d]]];
   val = play[n, d, s, a, b, ga, gb];
   nf = n - FE[[s + 1]];
   If[nf > 0,
    vf = If[gb == 1, v[nf, d, Which[s == 1, 1, s >= 3, If[nf > SC, 4, 1], True, 0], a, b, ga, 0],
      p3 = If[d <= -3, psf3, 0]; (1 - p3) f[2, nf, d, a, b, ga, 0] + p3 f[3, nf, d, a, b, ga, 0]];
    vf = (1 - ppre) vf + ppre Max[vf, early[n, d, s, a, b, ga, gb]];
    val = Min[val, vf]];
   If[s == 0 && a > 0, val = Max[val, v[n, d, 1, a - 1, b, ga, gb]]];
   val];

(* value of a k-point shot after tau steps *)
shot[n_, d_, s_, a_, b_, ga_, gb_, k_, tau_] := Module[{n2, n3, lost, mk, ms, sf, pm, px, cs},
   cs = If[s >= 4, 1, s]; n2 = n - tau;
   lost = If[n2 <= 0, term[d], 1 - v[n2, -d, 2, b, a, gb, ga]];
   If[n2 <= 0,
    mk = term[d + k]; ms = term[d]; sf = f[k, 0, d, a, b, ga, 0],
    mk = 1 - v[n2, -(d + k), 0, b, a, gb, ga]; sf = f[k, n2, d, a, b, ga, 0];
    n3 = n2 - RT;
    ms = miss[n3, d, a, b, ga, gb, rfg, thfg, etafg]];
   {pm, px} = If[k == 2, {M2[[cs + 1, tau]], X2[[cs + 1, tau]]}, {M3[[cs + 1, tau]], X3[[cs + 1, tau]]}];
   (ptov + NONE[[cs + 1, tau]]) lost + psf sf + pm mk + px ms];
early[n_, d_, s_, a_, b_, ga_, gb_] := Max[-1, Table[
    If[k == 2 && ALLOW2[[If[s >= 4, 1, s] + 1, tau]] == 0, -1, shot[n, d, s, a, b, ga, gb, k, tau]],
    {tau, 1, FE[[s + 1]] - 1}, {k, {2, 3}}]];

play[n_, d_, s_, a_, b_, ga_, gb_] := Module[{best, n2, n3, lost, mk, ms, sf, pm, px, nt, clock, cs, scl},
   (* where the remaining shot clock scl is shorter than the game clock it forces a shot within scl steps;
      holding is a violation that gives the opponent a dead ball *)
   scl = clk[s]; clock = (scl > 0 && n > scl); cs = If[s >= 4, 1, s];  (* FC14, FC14T use the FC curves *)
   best = If[clock, 1 - v[n - scl, -d, 0, b, a, gb, ga], term[d]];
   Do[
    n2 = n - tau;
    lost = If[n2 <= 0, term[d], 1 - v[n2, -d, 2, b, a, gb, ga]];
    Do[
     If[k == 2 && ALLOW2[[cs + 1, tau]] == 0, Continue[]];
     If[n2 <= 0,
      mk = term[d + k]; ms = term[d]; sf = f[k, 0, d, a, b, ga, 0],
      mk = 1 - v[n2, -(d + k), 0, b, a, gb, ga]; sf = f[k, n2, d, a, b, ga, 0];
      n3 = n2 - RT;
      ms = miss[n3, d, a, b, ga, gb, rfg, thfg, etafg]];
     {pm, px} = If[k == 2, {M2[[cs + 1, tau]], X2[[cs + 1, tau]]}, {M3[[cs + 1, tau]], X3[[cs + 1, tau]]}];
     best = Max[best, (ptov + NONE[[cs + 1, tau]]) lost + psf sf + pm mk + px ms],
     {k, {2, 3}}],
    {tau, 1, If[clock, scl, n]}];
   If[(s == 2 || s == 3) && a > 0,
    nt = n - CTO; best = Max[best, If[nt > 0, v[nt, d, If[s == 3 && n > SC, 5, 1], a - 1, b, ga, gb], term[d]]]];
   best];

(* free-throw trips: value of the shooting team with j FTs left *)
(* a kept rebound is live (OREB) or, with share thft / thdm, a dead-ball team rebound: frontcourt inbound *)
(* a missed FT or FG: a kept rebound is live (OR) or, with share th*, a dead-ball team rebound (frontcourt inbound,
   FC14 while the 14-s clock binds); a lost one is live (opponent LV) or, with share eta*, a dead-ball team rebound
   to the defence (the opponent inbounds in its backcourt, BC); FT miss types have their own delays *)
miss[n3_, d_, a_, b_, ga_, gb_, rho_, th_, eta_] := If[n3 <= 0, term[d],
   rho ((1 - th) v[n3, d, 3, a, b, ga, gb] + th v[n3, d, If[n3 > SC, 4, 1], a, b, ga, gb]) +
    (1 - rho) ((1 - eta) (1 - v[n3, -d, 2, b, a, gb, ga]) + eta (1 - v[n3, -d, 0, b, a, gb, ga]))];
fparts[n_, d_, a_, b_, ga_, gb_] := fparts[n, d, a, b, ga, gb] = Module[{mk, mso, msd},
   If[n == 0, mk = term[d + 1]; mso = msd = term[d],
    mk = 1 - v[n, -(d + 1), 0, b, a, gb, ga];
    mso = Sum[WO[[k]] miss[n - RTO[[k]], d, a, b, ga, gb, rft, thft, etaft], {k, Length[RTO]}];
    msd = Sum[WD[[k]] miss[n - RTD[[k]], d, a, b, ga, gb, rdm, thdm, etadm], {k, Length[RTD]}]];
   {q mk + (1 - q) mso, msd}];
f[j_, n_, d_, a_, b_, ga_, gb_] /; d =!= cl[d] := f[j, n, cl[d], a, b, ga, gb];
f[1, n_, d_, a_, b_, ga_, gb_] := Max @@ fparts[n, d, a, b, ga, gb];
f[j_, n_, d_, a_, b_, ga_, gb_] := f[j, n, d, a, b, ga, gb] =
   q f[j - 1, n, d + 1, a, b, ga, gb] + (1 - q) f[j - 1, n, d, a, b, ga, gb];

py = Import[FileNameJoin[{dir, "dp_small_python.json"}], "RawJSON"];
dv = Table[Abs[N[v @@ Most[row], 30] - Last[row]], {row, py["V"]}];
dft = Flatten@Table[Abs[N[fparts[row[[1]], row[[2]], row[[3]], row[[4]], 0, 0], 30] - row[[5 ;; 6]]], {row, py["FT"]}];
exampleExact = v[NS, -3, 0, 0, 0, 0, 0];

(* Part 2: last-FT break-even rebound rate *)
(* fresh symbols: q, rft and rdm above hold the fitted numbers, so the identity is proved for general
   symbols QS, RFTS, RDMS, MK, KDM, KFT, LDM, LFT (KDM, KFT: value of a kept deliberate / ordinary miss; LDM, LFT:
   value of a lost one; each at its own rebound delay and a mix of live and dead-ball rebounds) *)
Clear[QS, RFTS, RDMS, MK, KDM, KFT, LDM, LFT];
fmk = QS MK + (1 - QS) (RFTS KFT + (1 - RFTS) LFT);
fms = RDMS KDM + (1 - RDMS) LDM;
rstar = (QS (MK - LDM) + (1 - QS) (RFTS (KFT - LDM) + (1 - RFTS) (LFT - LDM)))/(KDM - LDM);
ident = FullSimplify[fms - fmk == (KDM - LDM) (RDMS - rstar), KDM != LDM];
(* with equal kept and equal lost values (KDM = KFT = KEEP, LDM = LFT = LOSE) it reduces to
   rho* = (1 - q) rho_ft + q (MK - LOSE)/(KEEP - LOSE) *)
identSame = FullSimplify[(rstar /. {KDM -> KEEP, KFT -> KEEP, LDM -> LOSE, LFT -> LOSE}) ==
    (1 - QS) RFTS + QS (MK - LOSE)/(KEEP - LOSE), KEEP != LOSE];
identFree = Union[Cases[{fmk, fms, rstar}, _Symbol?(Context[#] === "Global`" &), Infinity]];
ftj = Import[FileNameJoin[{dir, "ft_identity_python.json"}], "RawJSON"];
chk = Table[With[{rs = (ftj["q"] (w["mk"] - w["ldm"]) + (1 - ftj["q"]) (ftj["rho_ft"] (w["kft"] - w["ldm"]) +
        (1 - ftj["rho_ft"]) (w["lft"] - w["ldm"])))/(w["kdm"] - w["ldm"])},
    {Abs[(w["fms"] - w["fmk"]) - (w["kdm"] - w["ldm"]) (ftj["rho_dm"] - rs)],
     Sign[w["fms"] - w["fmk"]] == Sign[ftj["rho_dm"] - rs] || Abs[w["fms"] - w["fmk"]] < 10^-12}], {w, ftj["rows"]}];

res = <|"wolframVersion" -> $Version,
   "smallInstanceStates" -> Length[dv], "maxAbsDiffV" -> Max[dv],
   "ftStates" -> Length[dft], "maxAbsDiffFT" -> Max[dft],
   "exampleV_NS_down3_BC" -> ToString[InputForm[exampleExact]], "exampleV_N" -> N[exampleExact, 20],
   "breakEvenIdentity" -> ToString[ident], "reducesWhenKeepEqual" -> ToString[identSame],
   "identitySymbols" -> ToString[identFree],
   "breakEvenFormula" -> "rho* = (q (mk - l_dm) + (1-q) (rho_ft (k_ft - l_dm) + (1-rho_ft) (l_ft - l_dm)))/(k_dm - l_dm)",
   "pPreSmallInstance" -> ToString[ppre, InputForm], "rtoRtdSmallInstance" -> {RTO, RTD, ToString[WD, InputForm]},
   "fittedRows" -> Length[chk], "maxIdentityResidual" -> Max[chk[[All, 1]]],
   "signAgreementAll" -> And @@ chk[[All, 2]]|>;
Export[FileNameJoin[{dir, "dp_check_out.json"}], res, "JSON"];
Print[res];
If[Max[dv] > 10^-12 || Max[dft] > 10^-12 || ident =!= True || identSame =!= True || !(And @@ chk[[All, 2]]), Print["CHECK_FAILED"]; Exit[1]];
Print["ALL_CHECKS_PASSED"];
