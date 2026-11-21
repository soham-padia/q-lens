# Token readout of the swap jets (swap_tokens.json)

Vocabulary: 248077 tokens (ids < len(tokenizer)); centred logits. Layers [24, 40, 56]. Lists unfiltered.

## Gates (relative L2 over the centred vocabulary unless noted)
| swap | L | J vs FD | Q vs FD | J vs FD (Richardson) | Q vs FD (Richardson) | g1 vs S2 | g2 vs S2 | r0 vs S2 (abs) | z(0) vs HF logits | FD top-k overlap Q/2 up/down | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| capital | 24 | 2.7e-03 | 5.8e-03 | 2.5e-04 | 6.5e-03 | 1.7e-06 | 1.0e-05 | 1.2e-05 | 3.1e-06 | 1.0/1.0 | 1.8 |
| capital | 40 | 3.0e-03 | 7.0e-03 | 1.7e-04 | 5.7e-03 | 3.1e-07 | 0.0e+00 | 1.2e-05 | 3.1e-06 | 1.0/1.0 | 1.3 |
| capital | 56 | 1.5e-03 | 4.4e-03 | 9.2e-05 | 4.8e-03 | 0.0e+00 | 4.4e-07 | 1.2e-05 | 3.1e-06 | 1.0/1.0 | 0.3 |
| language | 24 | 2.1e-03 | 9.8e-03 | 3.7e-04 | 1.3e-02 | 4.0e-07 | 6.4e-07 | 4.7e-05 | 1.7e-06 | 1.0/1.0 | 1.5 |
| language | 40 | 4.7e-03 | 6.9e-03 | 1.7e-04 | 3.4e-03 | 6.3e-08 | 7.9e-08 | 4.7e-05 | 1.7e-06 | 1.0/1.0 | 0.9 |
| language | 56 | 9.0e-04 | 6.3e-03 | 8.7e-05 | 5.9e-03 | 1.3e-07 | 0.0e+00 | 4.7e-05 | 1.7e-06 | 1.0/1.0 | 0.3 |
| continent | 24 | 1.4e-03 | 1.6e-02 | 7.3e-04 | 1.9e-02 | 0.0e+00 | 4.3e-07 | 1.3e-05 | 2.0e-06 | 1.0/1.0 | 1.5 |
| continent | 40 | 3.4e-03 | 7.0e-03 | 1.9e-04 | 5.7e-03 | 0.0e+00 | 2.9e-07 | 1.3e-05 | 2.0e-06 | 1.0/1.0 | 0.9 |
| continent | 56 | 4.8e-04 | 1.0e-02 | 1.3e-04 | 1.3e-02 | 1.6e-07 | 0.0e+00 | 1.3e-05 | 2.0e-06 | 1.0/1.0 | 0.3 |

## Doses, sizes and agreement
| swap | L | λ* actual | λ* J+Q | λ* J | |J| | |Q/2| | cos(J, Q/2) | cos(pred, actual) at λ* | cos(λ*J, actual) | rel err pred | rel err J only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| capital | 24 | 1.12 | 13.62 | 37.76 | 169.4 | 167.9 | +0.83 | 0.70 | 0.64 | 0.75 | 0.79 |
| capital | 40 | 0.65 | 1.71 | 5.55 | 151.9 | 131.6 | +0.70 | 0.62 | 0.51 | 0.80 | 0.88 |
| capital | 56 | 0.84 | 1.05 | 1.92 | 200.3 | 129.8 | +0.59 | 0.92 | 0.85 | 0.39 | 0.56 |
| language | 24 | 1.06 | 3.03 | 10.78 | 77.4 | 69.4 | +0.83 | 0.70 | 0.61 | 0.74 | 0.86 |
| language | 40 | 0.67 | 0.98 | 2.50 | 168.6 | 202.9 | +0.82 | 0.80 | 0.70 | 0.62 | 0.78 |
| language | 56 | 0.92 | 0.88 | 1.25 | 187.4 | 90.8 | +0.47 | 0.91 | 0.90 | 0.46 | 0.44 |
| continent | 24 | 1.10 | 3.00 | 7.16 | 73.2 | 43.9 | +0.63 | 0.60 | 0.56 | 0.82 | 0.87 |
| continent | 40 | 0.68 | 0.91 | 1.99 | 133.3 | 140.5 | +0.79 | 0.82 | 0.72 | 0.60 | 0.77 |
| continent | 56 | 2.46 | 1.73 | 2.60 | 142.4 | 45.4 | +0.30 | 0.89 | 0.77 | 0.82 | 0.72 |

## Ranks of the answers (target = new: rank among the most raised; source = old: rank among the most lowered)
| swap | L | target ↑ J | ↑ Q/2 | ↑ J+Q/2 at λ* | ↑ actual at λ* | source ↓ J | ↓ Q/2 | ↓ J+Q/2 at λ* | ↓ actual |
|---|---|---|---|---|---|---|---|---|---|
| capital: France→China (Paris→Beijing) | 24 | 71402 | 88144 | 77888 | 59 | 90723 | 148341 | 120427 | 4 |
| capital: France→China (Paris→Beijing) | 40 | 49 | 1 | 9 | 1 | 44879 | 34132 | 36265 | 1355 |
| capital: France→China (Paris→Beijing) | 56 | 5 | 1 | 1 | 1 | 154618 | 206580 | 180688 | 169 |
| language: France→China (French→Chinese) | 24 | 358 | 1378 | 545 | 7 | 15206 | 2164 | 4938 | 4 |
| language: France→China (French→Chinese) | 40 | 10 | 2 | 4 | 1 | 1067 | 1246 | 806 | 11 |
| language: France→China (French→Chinese) | 56 | 2 | 1 | 1 | 2 | 19822 | 684 | 3656 | 23 |
| continent: France→China (Europe→Asia) | 24 | 1 | 79 | 5 | 1 | 214468 | 56853 | 167173 | 4483 |
| continent: France→China (Europe→Asia) | 40 | 2 | 3 | 2 | 2 | 79999 | 8217 | 32800 | 1186 |
| continent: France→China (Europe→Asia) | 56 | 56 | 4 | 25 | 40 | 36180 | 45071 | 25772 | 890 |

## Token lists (top 10; value = centred logit change per unit λ for J, per unit λ² for Q/2)
### capital: France→China (Paris→Beijing), L24  (λ* = 1.12)
- Q/2 raises: `Vi` +4.67, `␣vi` +4.54, `␣Vienna` +4.44, `维也纳` +4.41, `␣Vi` +4.41, `_vi` +3.97, `.V` +3.73, `␣ви` +3.73, `␣Vien` +3.70, `vi` +3.68
- Q/2 lowers: `␣LY` -1.90, `␣Fonse` -1.50, `␣Ly` -1.48, `_ly` -1.47, `␣Cuar` -1.45, `␣ly` -1.43, `ely` -1.42, `/*@` -1.41 [punct], `␣garg` -1.39, `␣Rit` -1.39
- J raises: `␣Vienna` +4.74, `维也纳` +4.40, `Vi` +4.25, `␣Vi` +3.87, `␣vi` +3.80, `␣Wien` +3.64, `␣Vien` +3.58, `␣Wiener` +3.57, `vi` +3.44, `␣Salzburg` +3.39
- J lowers: `␣garg` -1.63, `展览展示` -1.56, `凯旋` -1.51, `_evt` -1.51, `␣towers` -1.48, `/npm` -1.45, `aisons` -1.45, `␣champions` -1.43, `␣arr` -1.42, `␣Эй` -1.42
- actual change at λ* raises: `Viet` +6.14, `Vi` +5.72, `␣Vietnam` +5.69, `␣Bangkok` +5.67, `维也纳` +5.48, `␣Vi` +5.24, `Wi` +5.16, `Bang` +5.11, `␣Viet` +5.10, `␣Vienna` +5.08
- actual change at λ* lowers: `巴黎` -7.18, `␣paris` -6.86, `␣PARIS` -6.66, `␣Paris` -6.59, `␣파리` -6.13, `␣Пари` -6.12, `パリ` -5.87, `␣باريس` -5.65, `␣LY` -5.54, `␣PAR` -5.49

### capital: France→China (Paris→Beijing), L40  (λ* = 0.65)
- Q/2 raises: `␣Beijing` +2.20, `Viet` +1.97, `␣Китай` +1.96, `␣Vietnam` +1.92, `␣китай` +1.92, `␣Bangkok` +1.83, `␣الصينية` +1.82, `␣Zhao` +1.77, `␣Yang` +1.76, `␣Cina` +1.76
- Q/2 lowers: `lops` -1.31, `cps` -1.29, `tps` -1.24, `吕` -1.20, `␣André` -1.19, `ensor` -1.17, `spf` -1.16, `␣spar` -1.16, `␣fit` -1.15, `ütz` -1.10
- J raises: `␣Vien` +2.44, `维也纳` +2.43, `␣Vienna` +2.41, `␣Vietnam` +2.38, `越南` +2.32, `Vien` +2.22, `Viet` +2.15, `Vi` +2.10, `␣Pakistan` +2.09, `␣Viet` +2.01
- J lowers: `␣nc` -1.45, `␣Nantes` -1.43, `␣nymph` -1.42, `␣Arts` -1.40, `␣André` -1.38, `␣Abel` -1.38, `␣PROVINS` -1.37, `␣Ét` -1.37, `␣Lantern` -1.37, `␣Jules` -1.36
- actual change at λ* raises: `␣Beijing` +8.97, `␣Bangkok` +7.01, `␣Guang` +6.85, `␣Jiang` +6.32, `␣Jing` +6.26, `␣Tian` +6.26, `␣Bao` +6.20, `Be` +6.11, `␣Cheng` +6.09, `␣Beij` +6.06
- actual change at λ* lowers: `␣LY` -3.01, `瑕` -2.77, `airy` -2.75, `吕` -2.74, `␣Font` -2.74, `孚` -2.72, `␣spar` -2.69, `␣cds` -2.65, `␣Lyme` -2.64, `␣parsley` -2.63

### capital: France→China (Paris→Beijing), L56  (λ* = 0.84)
- Q/2 raises: `␣Beijing` +4.57, `Be` +4.38, `_be` +4.15, `␣BE` +3.99, `_BE` +3.98, `-be` +3.93, `.be` +3.82, `BE` +3.70, `␣Be` +3.69, `/be` +3.63
- Q/2 lowers: `ufs` -1.29, `␣revan` -1.12, `ussy` -1.12, `cly` -1.08, `␣arc` -1.06, `␣Cứ` -1.06, `␣tool` -1.06, `␣Arc` -1.05, `EDO` -1.05, `ARRY` -1.05
- J raises: `␣China` +6.71, `␣Chinese` +6.60, `China` +6.28, `Chinese` +6.18, `␣Beijing` +5.59, `␣china` +5.32, `␣chinese` +5.23, `china` +5.13, `␣Китай` +5.06, `-China` +5.06
- J lowers: `␣Bob` -2.12, `␣Fraction` -2.08, `␣frags` -2.04, `␣Fr` -2.03, `␣bob` -2.03, `bob` -1.97, `␣fr` -1.97, `VERS` -1.95, `/fr` -1.95, `␣Frans` -1.94
- actual change at λ* raises: `␣Beijing` +8.90, `␣Chinese` +8.07, `␣China` +8.01, `Chinese` +7.52, `China` +7.49, `␣chinese` +6.79, `china` +6.70, `␣china` +6.61, `␣Китай` +6.35, `␣китай` +6.34
- actual change at λ* lowers: `␣France` -2.75, `法国` -2.70, `VERS` -2.59, `␣파` -2.40, `France` -2.38, `ARRY` -2.35, `␣França` -2.29, `法兰` -2.25, `␣Vers` -2.23, `␣фран` -2.18

### language: France→China (French→Chinese), L24  (λ* = 1.06)
- Q/2 raises: `␣Bav` +1.30, `␣Austrian` +1.25, `␣Thai` +1.21, `Thai` +1.10, `奥地利` +1.09, `Hung` +1.09, `␣Augsburg` +1.08, `Dan` +1.05, `␣German` +1.05, `German` +1.04
- Q/2 lowers: `_FF` -0.77, `篷` -0.62, `inking` -0.62, `靓` -0.59, `␣france` -0.58, `eel` -0.58, `霞` -0.57, `ignon` -0.57, `olé` -0.57, `␣Î` -0.56
- J raises: `␣Thai` +1.29, `␣Austrian` +1.28, `Ar` +1.21, `␣Arabic` +1.21, `␣Asian` +1.19, `␣Bav` +1.19, `␣Austria` +1.16, `Thai` +1.15, `␣Tibetan` +1.12, `␣asi` +1.11
- J lowers: `eel` -0.72, `ezi` -0.70, `_rq` -0.68, `eze` -0.67, `_FF` -0.67, `ée` -0.66, `_PF` -0.65, `REFERRED` -0.65, `淇` -0.64, `enan` -0.64
- actual change at λ* raises: `␣Thai` +7.22, `Thai` +6.68, `泰` +6.37, `␣Canton` +5.58, `␣Asian` +5.48, `Asian` +5.40, `␣Chinese` +5.23, `␣Mandarin` +5.22, `␣Thailand` +5.20, `␣китай` +5.09
- actual change at λ* lowers: `法国` -4.93, `␣France` -4.80, `␣FRANCE` -4.53, `␣French` -4.41, `法國` -4.36, `法语` -4.35, `프` -4.33, `␣french` -4.31, `巴黎` -4.28, `法兰` -4.28

### language: France→China (French→Chinese), L40  (λ* = 0.67)
- Q/2 raises: `␣китай` +5.20, `␣Chinese` +5.04, `␣Mandarin` +4.92, `␣Китай` +4.68, `Chinese` +4.62, `␣Asian` +4.57, `Asian` +4.56, `␣chino` +4.45, `␣chinese` +4.40, `␣China` +4.40
- Q/2 lowers: `␣Bleu` -1.93, `alc` -1.81, `␣Roch` -1.81, `␣Romans` -1.80, `萝` -1.76, `␣occ` -1.76, `霞` -1.74, `罗纳` -1.74, `␣prov` -1.73, `␣Pro` -1.72
- J raises: `␣Asian` +3.57, `␣asi` +3.40, `Asian` +3.29, `␣Thai` +3.27, `␣китай` +3.16, `␣Asia` +3.15, `亚洲` +3.05, `Thai` +3.02, `Asia` +2.96, `␣Chinese` +2.95
- J lowers: `dain` -1.58, `eau` -1.54, `␣Roch` -1.52, `_FF` -1.46, `␣Bleu` -1.45, `法国` -1.44, `vain` -1.42, `riot` -1.39, `embl` -1.39, `eleg` -1.38
- actual change at λ* raises: `␣Chinese` +6.61, `␣китай` +6.48, `␣mand` +6.35, `␣Mandarin` +6.27, `␣Canton` +6.16, `Chinese` +6.13, `␣China` +6.11, `China` +5.91, `␣chino` +5.90, `␣mandar` +5.72
- actual change at λ* lowers: `프` -3.41, `法兰` -3.30, `␣Loire` -3.26, `法国` -3.22, `beau` -3.19, `␣Roch` -3.06, `␣France` -3.05, `␣FRANCE` -2.97, `␣Roque` -2.96, `法國` -2.92

### language: France→China (French→Chinese), L56  (λ* = 0.92)
- Q/2 raises: `␣Chinese` +3.16, `␣китай` +3.08, `␣chino` +2.93, `␣China` +2.93, `␣chines` +2.92, `Chinese` +2.91, `␣Zhang` +2.90, `␣Zhao` +2.88, `China` +2.88, `␣chinese` +2.86
- Q/2 lowers: `␣FC` -1.05, `␣فر` -0.97, `␣Friendly` -0.91, `␣Frances` -0.87, `␣Fro` -0.87, `␣Fred` -0.87, `␣Freak` -0.84, `enhagen` -0.83, `␣Fortal` -0.83, `␣FR` -0.82
- J raises: `␣China` +7.33, `␣Chinese` +7.13, `China` +6.96, `Chinese` +6.78, `␣chino` +6.30, `␣китай` +6.23, `␣중국` +5.98, `␣chinese` +5.95, `␣chines` +5.91, `␣china` +5.90
- J lowers: `␣France` -2.53, `France` -1.99, `␣FRANCE` -1.86, `␣bon` -1.70, `フ` -1.63, `OnError` -1.63, `␣Provence` -1.63, `rxjs` -1.62, `␣Î` -1.61, `委内` -1.60
- actual change at λ* raises: `␣China` +7.42, `␣Chinese` +6.95, `China` +6.88, `Chinese` +6.47, `␣chino` +6.40, `␣중국` +6.13, `␣китай` +6.12, `␣chines` +6.12, `␣china` +5.94, `-China` +5.91
- actual change at λ* lowers: `␣France` -4.22, `France` -3.49, `␣FRANCE` -3.41, `フ` -3.35, `法国` -3.31, `␣france` -3.11, `␣Frances` -3.01, `_FR` -2.93, `␣Франции` -2.91, `프` -2.91

### continent: France→China (Europe→Asia), L24  (λ* = 1.10)
- Q/2 raises: `奥地利` +0.74, `␣Zentral` +0.70, `␣Austrian` +0.69, `Central` +0.66, `␣Central` +0.66, `中央` +0.66, `␣Alpine` +0.64, `␣Alpen` +0.64, `␣Austria` +0.63, `␣central` +0.62
- Q/2 lowers: `␣Toulouse` -0.49, `␣Mercury` -0.47, `Med` -0.46, `␣afr` -0.46, `␣african` -0.46, `␣France` -0.45, `␣french` -0.45, `WL` -0.45, `France` -0.44, `␣france` -0.44
- J raises: `␣Asia` +1.21, `␣Swiss` +1.17, `␣alp` +1.14, `␣Switzerland` +1.14, `␣Alpine` +1.12, `␣asi` +1.09, `␣Asian` +1.07, `Asia` +1.03, `␣Austrian` +1.02, `␣Zurich` +1.01
- J lowers: `␣mét` -0.73, `西` -0.73, `西北` -0.72, `␣⇒` -0.66 [punct], `␣Caption` -0.66, `布` -0.65, `␣Ouest` -0.63, `␣wę` -0.62, `␣Filename` -0.62, `␣Completion` -0.62
- actual change at λ* raises: `␣Asia` +6.38, `␣asi` +6.20, `␣Asian` +5.91, `asia` +5.89, `␣Asians` +5.76, `Asia` +5.40, `_As` +5.39, `亚洲` +5.37, `_as` +5.37, `␣Asi` +5.30
- actual change at λ* lowers: `法国` -4.93, `␣France` -4.64, `epsilon` -4.29, `␣French` -4.29, `␣france` -4.23, `法语` -4.22, `France` -4.17, `␣프랑스` -4.09, `French` -4.04, `␣FRANCE` -4.04

### continent: France→China (Europe→Asia), L40  (λ* = 0.68)
- Q/2 raises: `␣asi` +4.74, `␣Asian` +4.58, `␣Asia` +4.45, `Asian` +4.27, `亚洲` +4.27, `Asia` +4.23, `asia` +4.19, `␣Asi` +4.02, `␣asiat` +3.97, `␣Asians` +3.94
- Q/2 lowers: `␣Western` -1.88, `西部` -1.75, `Western` -1.71, `western` -1.71, `西南部` -1.69, `␣western` -1.58, `(w` -1.48, `␣france` -1.43, `␣FRANCE` -1.42, `在西` -1.42
- J raises: `␣asi` +3.73, `␣Asia` +3.65, `␣Asian` +3.53, `Asia` +3.44, `亚洲` +3.33, `Asian` +3.26, `asia` +3.16, `␣asia` +3.10, `␣Asians` +3.08, `␣asiat` +2.97
- J lowers: `西部` -1.58, `Ben` -1.55, `␣Ben` -1.50, `西南部` -1.48, `␣Western` -1.44, `western` -1.43, `ben` -1.43, `西北` -1.40, `␣Northwestern` -1.36, `(w` -1.34
- actual change at λ* raises: `␣asi` +6.42, `␣Asia` +6.33, `␣Asian` +6.22, `asia` +6.18, `␣Asians` +5.83, `亚洲` +5.83, `Asia` +5.67, `␣asian` +5.67, `␣asiat` +5.53, `Asian` +5.48
- actual change at λ* lowers: `西部` -2.81, `␣Western` -2.79, `法兰` -2.52, `Western` -2.51, `.fr` -2.47, `␣France` -2.42, `␣FRANCE` -2.35, `␣Grenoble` -2.30, `␣Savo` -2.26, `western` -2.25

### continent: France→China (Europe→Asia), L56  (λ* = 2.46)
- Q/2 raises: `␣asi` +0.86, `␣Chang` +0.83, `asia` +0.80, `␣Asia` +0.78, `␣Asian` +0.77, `␣Jiang` +0.76, `␣Asi` +0.76, `␣Cheng` +0.73, `Asian` +0.73, `Asia` +0.72
- Q/2 lowers: `␣French` -1.65, `法国` -1.62, `French` -1.58, `␣France` -1.53, `France` -1.44, `epsilon` -1.41, `フランス` -1.40, `␣француз` -1.39, `␣epsilon` -1.32, `␣french` -1.31
- J raises: `␣China` +5.36, `China` +5.07, `␣Chinese` +5.03, `Chinese` +4.70, `␣china` +4.48, `␣chinese` +4.45, `-China` +4.44, `␣китай` +4.29, `␣chines` +4.20, `␣Китай` +4.16
- J lowers: `␣France` -4.14, `France` -3.77, `法国` -3.68, `法兰` -3.65, `.fr` -3.51, `␣FRANCE` -3.50, `␣france` -3.36, `Fr` -3.36, `␣franc` -3.34, `␣Fr` -3.30
- actual change at λ* raises: `␣China` +10.07, `␣Chinese` +9.07, `China` +8.87, `␣china` +8.48, `␣chinese` +8.20, `Chinese` +8.09, `-China` +7.94, `␣китай` +7.86, `␣chines` +7.75, `␣Китай` +7.59
- actual change at λ* lowers: `␣France` -12.87, `法国` -11.88, `France` -11.42, `␣French` -10.81, `␣france` -10.30, `␣FRANCE` -10.27, `French` -10.01, `フランス` -9.64, `␣프랑스` -9.44, `␣Франции` -9.22

## Junk count over all lists (classes: word / punct = no letter or digit / space = whitespace only / byte = undecodable)
- halfQ_up: word 90
- halfQ_down: punct 1, word 89
- J_up: word 90
- J_down: punct 1, word 89
- act_up: word 90
- act_down: word 90

## LaTeX (booktabs + tabularx; ␣ = \textvisiblespace; contains non-ASCII tokens: XeLaTeX/xeCJK or replace)
```latex
\begin{table*}[t]\centering\scriptsize
\caption{Tokens moved by the swap jets in logit space (centred), at the last position of the two-hop prompt, for the actual swap push at one layer. $\tfrac12 Q$: second-order term per unit $\lambda^2$; $J$: first-order term per unit $\lambda$. Ranks: target answer among the most raised, source answer among the most lowered, under $J$, $\tfrac12 Q$ and $\lambda^* J+\tfrac12\lambda^{*2}Q$ at the actual flip dose $\lambda^*$.}
\begin{tabularx}{\textwidth}{@{}l r r X X c c@{}}\toprule
Swap & $\ell$ & $\lambda^*$ & $\tfrac12 Q$ raises (top 10) & $\tfrac12 Q$ lowers (top 10) & target rank $J$/$\tfrac12Q$/$J{+}\tfrac12Q$ & source rank \\ \midrule
Paris$\to$Beijing & 24 & 1.12 & \texttt{Vi}, \texttt{\textvisiblespace{}vi}, \texttt{\textvisiblespace{}Vienna}, \texttt{维也纳}, \texttt{\textvisiblespace{}Vi}, \texttt{\_vi}, \texttt{.V}, \texttt{\textvisiblespace{}ви}, \texttt{\textvisiblespace{}Vien}, \texttt{vi} & \texttt{\textvisiblespace{}LY}, \texttt{\textvisiblespace{}Fonse}, \texttt{\textvisiblespace{}Ly}, \texttt{\_ly}, \texttt{\textvisiblespace{}Cuar}, \texttt{\textvisiblespace{}ly}, \texttt{ely}, \texttt{/*@}, \texttt{\textvisiblespace{}garg}, \texttt{\textvisiblespace{}Rit} & 71402/88144/77888 & 90723/148341/120427 \\
 & 40 & 0.65 & \texttt{\textvisiblespace{}Beijing}, \texttt{Viet}, \texttt{\textvisiblespace{}Китай}, \texttt{\textvisiblespace{}Vietnam}, \texttt{\textvisiblespace{}китай}, \texttt{\textvisiblespace{}Bangkok}, \texttt{\textvisiblespace{}الصينية}, \texttt{\textvisiblespace{}Zhao}, \texttt{\textvisiblespace{}Yang}, \texttt{\textvisiblespace{}Cina} & \texttt{lops}, \texttt{cps}, \texttt{tps}, \texttt{吕}, \texttt{\textvisiblespace{}André}, \texttt{ensor}, \texttt{spf}, \texttt{\textvisiblespace{}spar}, \texttt{\textvisiblespace{}fit}, \texttt{ütz} & 49/1/9 & 44879/34132/36265 \\
 & 56 & 0.84 & \texttt{\textvisiblespace{}Beijing}, \texttt{Be}, \texttt{\_be}, \texttt{\textvisiblespace{}BE}, \texttt{\_BE}, \texttt{-be}, \texttt{.be}, \texttt{BE}, \texttt{\textvisiblespace{}Be}, \texttt{/be} & \texttt{ufs}, \texttt{\textvisiblespace{}revan}, \texttt{ussy}, \texttt{cly}, \texttt{\textvisiblespace{}arc}, \texttt{\textvisiblespace{}Cứ}, \texttt{\textvisiblespace{}tool}, \texttt{\textvisiblespace{}Arc}, \texttt{EDO}, \texttt{ARRY} & 5/1/1 & 154618/206580/180688 \\
\midrule
French$\to$Chinese & 24 & 1.06 & \texttt{\textvisiblespace{}Bav}, \texttt{\textvisiblespace{}Austrian}, \texttt{\textvisiblespace{}Thai}, \texttt{Thai}, \texttt{奥地利}, \texttt{Hung}, \texttt{\textvisiblespace{}Augsburg}, \texttt{Dan}, \texttt{\textvisiblespace{}German}, \texttt{German} & \texttt{\_FF}, \texttt{篷}, \texttt{inking}, \texttt{靓}, \texttt{\textvisiblespace{}france}, \texttt{eel}, \texttt{霞}, \texttt{ignon}, \texttt{olé}, \texttt{\textvisiblespace{}Î} & 358/1378/545 & 15206/2164/4938 \\
 & 40 & 0.67 & \texttt{\textvisiblespace{}китай}, \texttt{\textvisiblespace{}Chinese}, \texttt{\textvisiblespace{}Mandarin}, \texttt{\textvisiblespace{}Китай}, \texttt{Chinese}, \texttt{\textvisiblespace{}Asian}, \texttt{Asian}, \texttt{\textvisiblespace{}chino}, \texttt{\textvisiblespace{}chinese}, \texttt{\textvisiblespace{}China} & \texttt{\textvisiblespace{}Bleu}, \texttt{alc}, \texttt{\textvisiblespace{}Roch}, \texttt{\textvisiblespace{}Romans}, \texttt{萝}, \texttt{\textvisiblespace{}occ}, \texttt{霞}, \texttt{罗纳}, \texttt{\textvisiblespace{}prov}, \texttt{\textvisiblespace{}Pro} & 10/2/4 & 1067/1246/806 \\
 & 56 & 0.92 & \texttt{\textvisiblespace{}Chinese}, \texttt{\textvisiblespace{}китай}, \texttt{\textvisiblespace{}chino}, \texttt{\textvisiblespace{}China}, \texttt{\textvisiblespace{}chines}, \texttt{Chinese}, \texttt{\textvisiblespace{}Zhang}, \texttt{\textvisiblespace{}Zhao}, \texttt{China}, \texttt{\textvisiblespace{}chinese} & \texttt{\textvisiblespace{}FC}, \texttt{\textvisiblespace{}فر}, \texttt{\textvisiblespace{}Friendly}, \texttt{\textvisiblespace{}Frances}, \texttt{\textvisiblespace{}Fro}, \texttt{\textvisiblespace{}Fred}, \texttt{\textvisiblespace{}Freak}, \texttt{enhagen}, \texttt{\textvisiblespace{}Fortal}, \texttt{\textvisiblespace{}FR} & 2/1/1 & 19822/684/3656 \\
\midrule
Europe$\to$Asia & 24 & 1.10 & \texttt{奥地利}, \texttt{\textvisiblespace{}Zentral}, \texttt{\textvisiblespace{}Austrian}, \texttt{Central}, \texttt{\textvisiblespace{}Central}, \texttt{中央}, \texttt{\textvisiblespace{}Alpine}, \texttt{\textvisiblespace{}Alpen}, \texttt{\textvisiblespace{}Austria}, \texttt{\textvisiblespace{}central} & \texttt{\textvisiblespace{}Toulouse}, \texttt{\textvisiblespace{}Mercury}, \texttt{Med}, \texttt{\textvisiblespace{}afr}, \texttt{\textvisiblespace{}african}, \texttt{\textvisiblespace{}France}, \texttt{\textvisiblespace{}french}, \texttt{WL}, \texttt{France}, \texttt{\textvisiblespace{}france} & 1/79/5 & 214468/56853/167173 \\
 & 40 & 0.68 & \texttt{\textvisiblespace{}asi}, \texttt{\textvisiblespace{}Asian}, \texttt{\textvisiblespace{}Asia}, \texttt{Asian}, \texttt{亚洲}, \texttt{Asia}, \texttt{asia}, \texttt{\textvisiblespace{}Asi}, \texttt{\textvisiblespace{}asiat}, \texttt{\textvisiblespace{}Asians} & \texttt{\textvisiblespace{}Western}, \texttt{西部}, \texttt{Western}, \texttt{western}, \texttt{西南部}, \texttt{\textvisiblespace{}western}, \texttt{(w}, \texttt{\textvisiblespace{}france}, \texttt{\textvisiblespace{}FRANCE}, \texttt{在西} & 2/3/2 & 79999/8217/32800 \\
 & 56 & 2.46 & \texttt{\textvisiblespace{}asi}, \texttt{\textvisiblespace{}Chang}, \texttt{asia}, \texttt{\textvisiblespace{}Asia}, \texttt{\textvisiblespace{}Asian}, \texttt{\textvisiblespace{}Jiang}, \texttt{\textvisiblespace{}Asi}, \texttt{\textvisiblespace{}Cheng}, \texttt{Asian}, \texttt{Asia} & \texttt{\textvisiblespace{}French}, \texttt{法国}, \texttt{French}, \texttt{\textvisiblespace{}France}, \texttt{France}, \texttt{epsilon}, \texttt{フランス}, \texttt{\textvisiblespace{}француз}, \texttt{\textvisiblespace{}epsilon}, \texttt{\textvisiblespace{}french} & 56/4/25 & 36180/45071/25772 \\
\bottomrule
\end{tabularx}\end{table*}

\begin{table*}[t]\centering\scriptsize
\caption{Contrast: tokens moved by the first-order term $J$ (same swaps and layers).}
\begin{tabularx}{\textwidth}{@{}l r X X@{}}\toprule
Swap & $\ell$ & $J$ raises (top 10) & $J$ lowers (top 10) \\ \midrule
Paris$\to$Beijing & 24 & \texttt{\textvisiblespace{}Vienna}, \texttt{维也纳}, \texttt{Vi}, \texttt{\textvisiblespace{}Vi}, \texttt{\textvisiblespace{}vi}, \texttt{\textvisiblespace{}Wien}, \texttt{\textvisiblespace{}Vien}, \texttt{\textvisiblespace{}Wiener}, \texttt{vi}, \texttt{\textvisiblespace{}Salzburg} & \texttt{\textvisiblespace{}garg}, \texttt{展览展示}, \texttt{凯旋}, \texttt{\_evt}, \texttt{\textvisiblespace{}towers}, \texttt{/npm}, \texttt{aisons}, \texttt{\textvisiblespace{}champions}, \texttt{\textvisiblespace{}arr}, \texttt{\textvisiblespace{}Эй} \\
 & 40 & \texttt{\textvisiblespace{}Vien}, \texttt{维也纳}, \texttt{\textvisiblespace{}Vienna}, \texttt{\textvisiblespace{}Vietnam}, \texttt{越南}, \texttt{Vien}, \texttt{Viet}, \texttt{Vi}, \texttt{\textvisiblespace{}Pakistan}, \texttt{\textvisiblespace{}Viet} & \texttt{\textvisiblespace{}nc}, \texttt{\textvisiblespace{}Nantes}, \texttt{\textvisiblespace{}nymph}, \texttt{\textvisiblespace{}Arts}, \texttt{\textvisiblespace{}André}, \texttt{\textvisiblespace{}Abel}, \texttt{\textvisiblespace{}PROVINS}, \texttt{\textvisiblespace{}Ét}, \texttt{\textvisiblespace{}Lantern}, \texttt{\textvisiblespace{}Jules} \\
 & 56 & \texttt{\textvisiblespace{}China}, \texttt{\textvisiblespace{}Chinese}, \texttt{China}, \texttt{Chinese}, \texttt{\textvisiblespace{}Beijing}, \texttt{\textvisiblespace{}china}, \texttt{\textvisiblespace{}chinese}, \texttt{china}, \texttt{\textvisiblespace{}Китай}, \texttt{-China} & \texttt{\textvisiblespace{}Bob}, \texttt{\textvisiblespace{}Fraction}, \texttt{\textvisiblespace{}frags}, \texttt{\textvisiblespace{}Fr}, \texttt{\textvisiblespace{}bob}, \texttt{bob}, \texttt{\textvisiblespace{}fr}, \texttt{VERS}, \texttt{/fr}, \texttt{\textvisiblespace{}Frans} \\
\midrule
French$\to$Chinese & 24 & \texttt{\textvisiblespace{}Thai}, \texttt{\textvisiblespace{}Austrian}, \texttt{Ar}, \texttt{\textvisiblespace{}Arabic}, \texttt{\textvisiblespace{}Asian}, \texttt{\textvisiblespace{}Bav}, \texttt{\textvisiblespace{}Austria}, \texttt{Thai}, \texttt{\textvisiblespace{}Tibetan}, \texttt{\textvisiblespace{}asi} & \texttt{eel}, \texttt{ezi}, \texttt{\_rq}, \texttt{eze}, \texttt{\_FF}, \texttt{ée}, \texttt{\_PF}, \texttt{REFERRED}, \texttt{淇}, \texttt{enan} \\
 & 40 & \texttt{\textvisiblespace{}Asian}, \texttt{\textvisiblespace{}asi}, \texttt{Asian}, \texttt{\textvisiblespace{}Thai}, \texttt{\textvisiblespace{}китай}, \texttt{\textvisiblespace{}Asia}, \texttt{亚洲}, \texttt{Thai}, \texttt{Asia}, \texttt{\textvisiblespace{}Chinese} & \texttt{dain}, \texttt{eau}, \texttt{\textvisiblespace{}Roch}, \texttt{\_FF}, \texttt{\textvisiblespace{}Bleu}, \texttt{法国}, \texttt{vain}, \texttt{riot}, \texttt{embl}, \texttt{eleg} \\
 & 56 & \texttt{\textvisiblespace{}China}, \texttt{\textvisiblespace{}Chinese}, \texttt{China}, \texttt{Chinese}, \texttt{\textvisiblespace{}chino}, \texttt{\textvisiblespace{}китай}, \texttt{\textvisiblespace{}중국}, \texttt{\textvisiblespace{}chinese}, \texttt{\textvisiblespace{}chines}, \texttt{\textvisiblespace{}china} & \texttt{\textvisiblespace{}France}, \texttt{France}, \texttt{\textvisiblespace{}FRANCE}, \texttt{\textvisiblespace{}bon}, \texttt{フ}, \texttt{OnError}, \texttt{\textvisiblespace{}Provence}, \texttt{rxjs}, \texttt{\textvisiblespace{}Î}, \texttt{委内} \\
\midrule
Europe$\to$Asia & 24 & \texttt{\textvisiblespace{}Asia}, \texttt{\textvisiblespace{}Swiss}, \texttt{\textvisiblespace{}alp}, \texttt{\textvisiblespace{}Switzerland}, \texttt{\textvisiblespace{}Alpine}, \texttt{\textvisiblespace{}asi}, \texttt{\textvisiblespace{}Asian}, \texttt{Asia}, \texttt{\textvisiblespace{}Austrian}, \texttt{\textvisiblespace{}Zurich} & \texttt{\textvisiblespace{}mét}, \texttt{西}, \texttt{西北}, \texttt{\textvisiblespace{}⇒}, \texttt{\textvisiblespace{}Caption}, \texttt{布}, \texttt{\textvisiblespace{}Ouest}, \texttt{\textvisiblespace{}wę}, \texttt{\textvisiblespace{}Filename}, \texttt{\textvisiblespace{}Completion} \\
 & 40 & \texttt{\textvisiblespace{}asi}, \texttt{\textvisiblespace{}Asia}, \texttt{\textvisiblespace{}Asian}, \texttt{Asia}, \texttt{亚洲}, \texttt{Asian}, \texttt{asia}, \texttt{\textvisiblespace{}asia}, \texttt{\textvisiblespace{}Asians}, \texttt{\textvisiblespace{}asiat} & \texttt{西部}, \texttt{Ben}, \texttt{\textvisiblespace{}Ben}, \texttt{西南部}, \texttt{\textvisiblespace{}Western}, \texttt{western}, \texttt{ben}, \texttt{西北}, \texttt{\textvisiblespace{}Northwestern}, \texttt{(w} \\
 & 56 & \texttt{\textvisiblespace{}China}, \texttt{China}, \texttt{\textvisiblespace{}Chinese}, \texttt{Chinese}, \texttt{\textvisiblespace{}china}, \texttt{\textvisiblespace{}chinese}, \texttt{-China}, \texttt{\textvisiblespace{}китай}, \texttt{\textvisiblespace{}chines}, \texttt{\textvisiblespace{}Китай} & \texttt{\textvisiblespace{}France}, \texttt{France}, \texttt{法国}, \texttt{法兰}, \texttt{.fr}, \texttt{\textvisiblespace{}FRANCE}, \texttt{\textvisiblespace{}france}, \texttt{Fr}, \texttt{\textvisiblespace{}franc}, \texttt{\textvisiblespace{}Fr} \\
\bottomrule
\end{tabularx}\end{table*}
```
