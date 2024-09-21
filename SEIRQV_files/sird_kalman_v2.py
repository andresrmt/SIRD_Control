from cycler import cycler
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from random import choices
from matplotlib import gridspec as Gs
import math
import random
from filterpy.kalman import KalmanFilter
import pandas as pd
from numpy.linalg import inv

###

import ipywidgets as widgets

###

#csse starts at 22 jan 20
#co.vid19.sg starts at 21 jan 20

#confirmed - csse
#deaths - csse
#recovered - co.vid19.sg
#recovered_csse = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 9, 15, 15, 17, 18, 18, 24, 29, 34, 34, 37, 37, 51, 51, 53, 62, 62, 62, 72, 72, 78, 78, 78, 78, 78, 78, 78, 78, 78, 96, 96, 97, 105, 105, 109, 114, 114, 114, 124, 140, 144, 144, 156, 160, 172, 183, 198, 212, 228, 240, 245, 266, 282, 297, 320, 344, 377, 406, 460, 492, 528, 560, 586, 611, 652, 683, 708, 740, 768, 801, 839, 896, 924, 956, 1002, 1060, 1095, 1128, 1188, 1244, 1268, 1347, 1408, 1457, 1519, 1634, 1712, 2040, 2296, 2721, 3225, 3851, 4809, 5973, 7248, 8342, 9340, 9835, 10365, 11207, 12117, 12995, 13882, 14876, 15738, 16444, 17276, 18294, 19631, 20727, 21699, 22466, 23175, 23582, 23904, 24209, 24559, 24886, 25368, 25877, 26532, 27286, 28040, 28808, 29589, 30366, 31163, 31938, 32712, 33459, 34224, 34942, 35590, 35995, 36299, 36604, 36825, 37163, 37508, 37985, 38500, 39011, 39429, 39769, 40117, 40441, 40717, 41002, 41323, 41645, 41780, 42026, 42285, 42541, 42737, 42988, 43256, 43577, 43833, 44086, 44371, 44584, 44795, 45015, 45172, 45352, 45521, 45692, 45893, 46098, 46308, 46491, 46740, 46926, 47179, 47454, 47768, 48031, 48312, 48583, 48915, 49609, 50128, 50520, 50736, 51049, 51521, 51953, 52350, 52533, 52810, 53119, 53651, 53920, 54164, 54587, 54816, 54971, 55139, 55337, 55447, 55586, 55658, 55749, 55891, 56028, 56174, 56267, 56333, 56408, 56461, 56492, 56558, 56607, 56699, 56764, 56802, 56884, 56955, 57039, 57071, 57142, 57181, 57241, 57262, 57291, 57333, 57341, 57359, 57367, 57393, 57466, 57488, 57512, 57534, 57562, 57575, 57597, 57612, 57624, 57668, 57675, 57698, 57705, 57728, 57740, 57752, 57764, 57784, 57798, 57807, 57819, 57819, 57821, 57829, 57832, 57844, 57858, 57879, 57883, 57890, 57899, 57909, 57913, 57924, 57924, 57937, 57938, 57949, 57959, 57968, 57975, 57981, 57985, 57990, 58002, 58008, 58019, 58029, 58033, 58039, 58046, 58052, 58058, 58064, 58067, 58071, 58079, 58091, 58104, 58111, 58119, 58124, 58134, 58139, 58144, 58145, 58152, 58158, 58160, 58168, 58176, 58182, 58188, 58192, 58197, 58208, 58210, 58233, 58238, 58252, 58265, 58274, 58279, 58287, 58304, 58322, 58332, 58352, 58362, 58370, 58386, 58400, 58411, 58449, 58449, 58476, 58487, 58497, 58517, 58541, 58562, 58580, 58611, 58636, 58668, 58694, 58722, 58757, 58771, 58784, 58846, 58868, 58894, 58926, 58959, 58983, 59015, 59041, 59066, 59086, 59104, 59148, 59181, 59196, 59228, 59271, 59301, 59320, 59348, 59373, 59405, 59433, 59484, 59506, 59526, 59558, 59569, 59604, 59621, 59641, 59661, 59676, 59679, 59697, 59719, 59731, 59746, 59753, 59761, 59785, 59803, 59816, 59823, 59830, 59842, 59849, 59857, 59870, 59879, 59894, 59900, 59905, 59911, 59939, 59950, 59961, 59968, 59974, 59984, 60001, 60014, 60019, 60022, 60038, 60051, 60063, 60078, 60086, 60103, 60113, 60122, 60131, 60138, 60149, 60161, 60176, 60185, 60202, 60214, 60239, 60260, 60284, 60304, 60322, 60335, 60357, 60374, 60392, 60417, 60446, 60463, 60485, 60503, 60540, 60576, 60603, 60613, 60629, 60662, 60682, 60704, 60718, 60718, 60751, 60765, 60786, 60806, 60823, 60844, 60873, 60906, 60912, 60933, 60953, 60975, 61006, 61029, 61047, 61062, 61104, 61123, 61134, 61183, 61229, 61242, 61277, 61294, 61316, 61329, 61360, 61372, 61407, 61423, 61434, 61459, 61481, 61523, 61557, 61580, 61613, 61635, 61660, 61702, 61740, 61765, 61799, 61799, 61869, 61894, 61911, 61931, 61960, 61987, 62023, 62042, 62070, 62098, 62113, 62140, 62161, 62181, 62195, 62212, 62563, 62228, 62234, 62250, 62265, 62286, 62299, 62313, 62341, 62363, 62374, 62397, 62414, 62432, 62453, 62467, 62481, 62498, 62512, 62526, 62532, 62543, 62560, 62576, 62587, 62595, 62605, 62617, 62637, 62663, 62679, 62733, 62863, 62957, 63033, 63252, 63357, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

N = 5850000

confirmed = np.array([0, 1, 3, 3, 4, 5, 7, 7, 10, 13, 16, 18, 18, 24, 28, 28, 30, 33, 40, 45, 47, 50, 58, 67, 72, 75, 77, 81, 84, 84, 85, 85, 89, 89, 91, 93, 93, 93, 102, 106, 108, 110, 110, 117, 130, 138, 150, 150, 160, 178, 178, 200, 212, 226, 243, 266, 313, 345, 385, 432, 455, 509, 558, 631, 683, 732, 802, 844, 879, 926, 1000, 1049, 1114, 1189, 1309, 1375, 1481, 1623, 1910, 2108, 2299, 2532, 2918, 3252, 3699, 4427, 5050, 5992, 6588, 8014, 9125, 10141, 11178, 12075, 12693, 13624, 14423, 14951, 15641, 16169, 17101, 17548, 18205, 18778, 19410, 20198, 20939, 21707, 22460, 23336, 23822, 24671, 25346, 26098, 26891, 27356, 28038, 28343, 28794, 29364, 29812, 30426, 31068, 31616, 31960, 32343, 32876, 33249, 33860, 34366, 34884, 35292, 35836, 36405, 36922, 37183, 37527, 37910, 38296, 38514, 38965, 39387, 39850, 40197, 40604, 40818, 40969, 41216, 41473, 41615, 41833, 42095, 42313, 42432, 42623, 42736, 42955, 43246, 43459, 43661, 43907, 44122, 44310, 44479, 44664, 44800, 44983, 45140, 45298, 45423, 45614, 45783, 45961, 46283, 46630, 46878, 47126, 47453, 47655, 47912, 48035, 48434, 48744, 49098, 49375, 49888, 50369, 50838, 51197, 51531, 51809, 52205, 52512, 52825, 53051, 53346, 54254, 54555, 54797, 54929, 55104, 55292, 55353, 55395, 55497, 55580, 55661, 55747, 55838, 55938, 56031, 56099, 56216, 56266, 56353, 56404, 56435, 56495, 56572, 56666, 56717, 56771, 56812, 56852, 56860, 56908, 56948, 56982, 57022, 57044, 57091, 57166, 57229, 57315, 57357, 57406, 57454, 57488, 57514, 57532, 57543, 57558, 57576, 57606, 57627, 57639, 57654, 57665, 57685, 57700, 57715, 57742, 57765, 57784, 57794, 57800, 57812, 57819, 57830, 57840, 57849, 57859, 57866, 57876, 57880, 57884, 57889, 57892, 57901, 57904, 57911, 57915, 57921, 57933, 57941, 57951, 57965, 57970, 57973, 57980, 57987, 57994, 58003, 58015, 58019, 58020, 58029, 58036, 58043, 58047, 58054, 58056, 58064, 58073, 58091, 58102, 58114, 58116, 58119, 58124, 58130, 58135, 58139, 58143, 58148, 58160, 58165, 58183, 58190, 58195, 58199, 58205, 58213, 58218, 58228, 58230, 58239, 58242, 58255, 58260, 58273, 58285, 58291, 58297, 58305, 58313, 58320, 58325, 58341, 58353, 58377, 58386, 58403, 58422, 58432, 58461, 58482, 58495, 58509, 58519, 58524, 58529, 58542, 58569, 58599, 58629, 58662, 58697, 58721, 58749, 58780, 58813, 58836, 58865, 58907, 58929, 58946, 58984, 59029, 59059, 59083, 59113, 59127, 59157, 59197, 59235, 59250, 59260, 59308, 59352, 59366, 59391, 59425, 59449, 59507, 59536, 59565, 59584, 59602, 59624, 59649, 59675, 59699, 59721, 59732, 59747, 59759, 59777, 59786, 59800, 59809, 59810, 59821, 59832, 59846, 59858, 59869, 59879, 59883, 59890, 59900, 59913, 59925, 59936, 59948, 59956, 59979, 59998, 60007, 60020, 60033, 60046, 60052, 60062, 60070, 60080, 60088, 60105, 60117, 60128, 60137, 60152, 60167, 60184, 60196, 60208, 60221, 60236, 60253, 60265, 60288, 60300, 60321, 60347, 60381, 60407, 60450, 60468, 60478, 60495, 60519, 60554, 60575, 60601, 60633, 60653, 60678, 60692, 60719, 60735, 60769, 60808, 60831, 60851, 60865, 60880, 60904, 60943, 60966, 61006, 61051, 61063, 61086, 61121, 61145, 61179, 61218, 61235, 61252, 61268, 61286, 61311, 61331, 61359, 61378, 61403, 61419, 61453, 61505, 61536, 61585, 61613, 61651, 61689, 61730, 61770, 61799, 61824, 61860, 61890, 61916, 61940, 61970, 62003, 62028, 62051, 62069, 62100, 62145, 62158, 62176, 62196, 62210, 62219, 62223, 62236, 62245, 62263, 62276, 62301, 62315, 62339, 62366, 62382, 62403, 62414, 62430, 62448, 62470, 62493, 62513, 62530, 62544, 62553, 62563, 62579, 62589, 62599, 62606, 62617, 62630, 62640, 62652, 62668, 62678, 62684, 62692, 62718, 62744, 62804, 62852, 62913, 62981, 63073, 63245, 63440, 63621, 63791, 63924, 64054, 64179, 64314, 64453, 64589, 64722, 64861, 64981, 65102, 65213, 65315, 65410, 65508, 65605, 65686, 65764, 65836, 65836, 65953, 66012, 66061, 66119, 66172, 66225, 66281, 66334, 66366, 66406, 66443, 66478, 66576, 66692, 66812, 66928, 66928, 67171, 67171, 67459, 67620, 67800, 67991, 68210, 68469, 68660, 68901, 69233, 69582, 70039, 70612, 71167, 71687, 72294, 73131, 73938, 74848, 75783, 76792, 77804, 78721, 79899, 81356, 82860, 84510, 85953, 87892, 89539, 91775, 94043, 96521, 99430, 101786, 103843, 106318, 109804, 113381, 116864, 120454, 124157, 126966, 129229, 132205, 135395, 138327, 141772, 145120, 148178, 150731, 154725, 158587, 162026, 165663, 169261, 172644, 175818, 179095, 184419, 187851, 192099, 195211, 198374, 200844, 204340, 207975, 210978, 212745, 215780, 218333, 220803, 224200, 227681, 230077, 233176, 235480, 237203, 239272, 241341, 244815, 244815, 248587, 250518, 252188, 253649, 255431, 257510, 258785, 259875, 261636, 262383, 263486, 264725, 266049, 267150, 267916, 268659, 269211, 269873, 270588, 271297, 271979, 272433, 272992, 273362, 273701, 274143, 274617, 274972, 275384, 275655, 275910, 276105, 276385, 276720, 277042, 277307, 277555, 277764, 278044, 278409, 278750, 279061, 279405, 279861, 280290, 280754, 281596, 282401, 283214, 283214, 284802, 285647, 286397, 287243, 288125, 289085, 290030])
deaths = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6, 6, 6, 6, 6, 6, 7, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 14, 14, 14, 15, 16, 17, 18, 18, 18, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 39, 40, 41, 42, 42, 42, 42, 43, 43, 44, 44, 44, 44, 45, 46, 46, 47, 47, 49, 50, 50, 52, 52, 52, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 57, 58, 58, 58, 58, 58, 58, 59, 59, 60, 60, 62, 65, 68, 70, 73, 76, 78, 80, 85, 93, 95, 103, 107, 113, 121, 130, 133, 136, 142, 153, 162, 172, 183, 192, 207, 215, 224, 233, 239, 246, 264, 280, 294, 300, 315, 329, 339, 349, 364, 380, 394, 407, 421, 430, 442, 459, 468, 480, 497, 511, 523, 540, 548, 562, 576, 586, 594, 612, 619, 619, 641, 654, 662, 667, 672, 678, 681, 684, 690, 701, 710, 718, 726, 735, 744, 746, 759, 763, 771, 774, 779, 783, 789, 794, 798, 804, 807, 808, 809, 810, 813, 815, 817, 818, 820, 820, 821, 822, 825, 825, 826, 827, 828, 829, 829, 829, 832, 834, 835, 835, 837, 838, 838, 838, 839, 839, 840])
recovered = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 9.0, 15.0, 15.0, 17.0, 18.0, 18.0, 24.0, 29.0, 34.0, 34.0, 37.0, 37.0, 51.0, 51.0, 53.0, 62.0, 62.0, 62.0, 72.0, 72.0, 78.0, 78.0, 78.0, 78.0, 78.0, 78.0, 78.0, 78.0, 78.0, 96.0, 96.0, 97.0, 105.0, 105.0, 109.0, 114.0, 114.0, 114.0, 124.0, 140.0, 144.0, 144.0, 156.0, 160.0, 172.0, 183.0, 198.0, 212.0, 228.0, 240.0, 245.0, 266.0, 282.0, 297.0, 320.0, 344.0, 377.0, 406.0, 460.0, 492.0, 528.0, 560.0, 586.0, 611.0, 652.0, 683.0, 708.0, 740.0, 768.0, 801.0, 839.0, 896.0, 924.0, 956.0, 1002.0, 1060.0, 1095.0, 1128.0, 1188.0, 1244.0, 1268.0, 1347.0, 1408.0, 1457.0, 1519.0, 1634.0, 1712.0, 2040.0, 2296.0, 2721.0, 3225.0, 3851.0, 4809.0, 5973.0, 7248.0, 8342.0, 9340.0, 9835.0, 10365.0, 11207.0, 12117.0, 12995.0, 13882.0, 14876.0, 15738.0, 16444.0, 17276.0, 18294.0, 19631.0, 20727.0, 21699.0, 22466.0, 23175.0, 23582.0, 23904.0, 24209.0, 24559.0, 24886.0, 25368.0, 25877.0, 26532.0, 27286.0, 28040.0, 28808.0, 29589.0, 30366.0, 31163.0, 31938.0, 32712.0, 33459.0, 34224.0, 34942.0, 35590.0, 35995.0, 36299.0, 36604.0, 36825.0, 37163.0, 37508.0, 37985.0, 38500.0, 39011.0, 39429.0, 39769.0, 40117.0, 40441.0, 40717.0, 41002.0, 41323.0, 41645.0, 41780.0, 42026.0, 42285.0, 42541.0, 42737.0, 42988.0, 43256.0, 43577.0, 43833.0, 44086.0, 44371.0, 44584.0, 44795.0, 45015.0, 45172.0, 45352.0, 45521.0, 45692.0, 45893.0, 46098.0, 46308.0, 46491.0, 46740.0, 46926.0, 47179.0, 47454.0, 47768.0, 48031.0, 48312.0, 48583.0, 48915.0, 49609.0, 50128.0, 50520.0, 50736.0, 51049.0, 51521.0, 51953.0, 52350.0, 52533.0, 52810.0, 53119.0, 53651.0, 53920.0, 54164.0, 54587.0, 54816.0, 54971.0, 55139.0, 55337.0, 55447.0, 55586.0, 55658.0, 55749.0, 55891.0, 56028.0, 56174.0, 56267.0, 56333.0, 56408.0, 56461.0, 56492.0, 56558.0, 56607.0, 56699.0, 56764.0, 56802.0, 56884.0, 56955.0, 57039.0, 57071.0, 57142.0, 57181.0, 57241.0, 57262.0, 57291.0, 57333.0, 57341.0, 57359.0, 57367.0, 57393.0, 57466.0, 57488.0, 57512.0, 57534.0, 57562.0, 57575.0, 57597.0, 57612.0, 57624.0, 57668.0, 57675.0, 57698.0, 57705.0, 57728.0, 57740.0, 57752.0, 57764.0, 57784.0, 57798.0, 57807.0, 57819.0, 57819.0, 57821.0, 57829.0, 57832.0, 57844.0, 57858.0, 57879.0, 57883.0, 57890.0, 57899.0, 57909.0, 57913.0, 57924.0, 57924.0, 57937.0, 57938.0, 57949.0, 57959.0, 57968.0, 57975.0, 57981.0, 57985.0, 57990.0, 58002.0, 58008.0, 58019.0, 58029.0, 58033.0, 58039.0, 58046.0, 58052.0, 58058.0, 58064.0, 58067.0, 58071.0, 58079.0, 58091.0, 58104.0, 58111.0, 58119.0, 58124.0, 58134.0, 58139.0, 58144.0, 58145.0, 58152.0, 58158.0, 58160.0, 58168.0, 58176.0, 58182.0, 58188.0, 58192.0, 58197.0, 58208.0, 58210.0, 58233.0, 58238.0, 58252.0, 58265.0, 58274.0, 58279.0, 58287.0, 58304.0, 58322.0, 58332.0, 58352.0, 58362.0, 58370.0, 58386.0, 58400.0, 58411.0, 58449.0, 58449.0, 58476.0, 58487.0, 58497.0, 58517.0, 58541.0, 58562.0, 58580.0, 58611.0, 58636.0, 58668.0, 58694.0, 58722.0, 58757.0, 58771.0, 58784.0, 58846.0, 58868.0, 58894.0, 58926.0, 58959.0, 58983.0, 59015.0, 59041.0, 59066.0, 59086.0, 59104.0, 59148.0, 59181.0, 59196.0, 59228.0, 59271.0, 59301.0, 59320.0, 59348.0, 59373.0, 59405.0, 59433.0, 59484.0, 59506.0, 59526.0, 59558.0, 59569.0, 59604.0, 59621.0, 59641.0, 59661.0, 59676.0, 59679.0, 59697.0, 59719.0, 59731.0, 59731.0, 59753.0, 59761.0, 59785.0, 59803.0, 59816.0, 59823.0, 59830.0, 59842.0, 59849.0, 59857.0, 59870.0, 59879.0, 59894.0, 59900.0, 59905.0, 59911.0, 59939.0, 59950.0, 59961.0, 59968.0, 59974.0, 59984.0, 60001.0, 60014.0, 60019.0, 60022.0, 60038.0, 60051.0, 60063.0, 60078.0, 60086.0, 60103.0, 60113.0, 60122.0, 60131.0, 60138.0, 60149.0, 60161.0, 60176.0, 60185.0, 60202.0, 60214.0, 60239.0, 60260.0, 60284.0, 60304.0, 60322.0, 60335.0, 60357.0, 60374.0, 60392.0, 60417.0, 60446.0, 60463.0, 60485.0, 60503.0, 60540.0, 60576.0, 60576.0, 60613.0, 60629.0, 60662.0, 60682.0, 60704.0, 60718.0, 60718.0, 60751.0, 60765.0, 60786.0, 60806.0, 60823.0, 60844.0, 60873.0, 60906.0, 60912.0, 60933.0, 60953.0, 60975.0, 61006.0, 61029.0, 61047.0, 61062.0, 61104.0, 61123.0, 61134.0, 61183.0, 61229.0, 61242.0, 61277.0, 61294.0, 61316.0, 61329.0, 61360.0, 61372.0, 61407.0, 61423.0, 61434.0, 61459.0, 61481.0, 61523.0, 61557.0, 61580.0, 61613.0, 61635.0, 61660.0, 61702.0, 61740.0, 61765.0, 61799.0, 61838.0, 61869.0, 61894.0, 61911.0, 61931.0, 61960.0, 61987.0, 62023.0, 62042.0, 62070.0, 62098.0, 62113.0, 62140.0, 62161.0, 62181.0, 62195.0, 62212.0, 62212.0, 62228.0, 62234.0, 62250.0, 62265.0, 62286.0, 62299.0, 62313.0, 62341.0, 62363.0, 62374.0, 62397.0, 62414.0, 62432.0, 62453.0, 62467.0, 62481.0, 62498.0, 62512.0, 62526.0, 62526.0, 62543.0, 62560.0, 62576.0, 62587.0, 62595.0, 62595.0, 62605.0, 62637.0, 62663.0, 62679.0, 62733.0, 62863.0, 62957.0, 63033.0, 63252.0, 63357.0, 63457.0, 63536.0, 63658.0, 63658.0, 64062.0, 64152.0, 64293.0, 64380.0, 64537.0, 64669.0, 64792.0, 64911.0, 65062.0, 65152.0, 65242.0, 65402.0, 65528.0, 65601.0, 65700.0, 65700.0, 65825.0, 65909.0, 65968.0, 66022.0, 66092.0, 66092.0, 66174.0, 66174.0, 66222.0, 66312.0, 66512.0, 66600.0, 66600.0, 66742.0, 67212.0, 67212.0, 67312.0, 67988.0, 68188.0, 68188.0, 68600.0, 68877.0, 69251.0, 69614.0, 70124.0, 70600.0, 70600.0, 71628.0, 72090.0, 72411.0, 73395.0, 74135.0, 74652.0, 74652.0, 76221.0, 77307.0, 78174.0, 79124.0, 80727.0, 81926.0, 84466.0, 86211.0, 86211.0, 92555.0, 92555.0, 98028.0, 98028.0, 100497.0, 104856.0, 104856.0, 108071.0, 108071.0, 108071.0, 108071.0, 108071.0, 108071.0, 108071.0, 110677.0, 113502.0, 117014.0, 120383.0, 123337.0, 123337.0, 126509.0, 130857.0, 133868.0, 133868.0, 141539.0, 145091.0, 145091.0, 148097.0, 151437.0, 160181.0, 164052.0, 166978.0, 166978.0, 173028.0, 175523.0, 175523.0, 182912.0, 185450.0, 187653.0, 187653.0, 193284.0, 193284.0, 195839.0, 202485.0, 202485.0, 207658.0, 207658.0, 209785.0, 213838.0, 217061.0, 217061.0, 221191.0, 223252.0, 225063.0, 226462.0, 227997.0, 230053.0, 231446.0, 232967.0, 234485.0, 235683.0, 236567.0, 237680.0, 239154.0, 240171.0, 241127.0, 241957.0, 242626.0, 243229.0, 243865.0, 244712.0, 245282.0, 245783.0, 246286.0, 246682.0, 247110.0, 247604.0, 247604.0, 248410.0, 248789.0, 249133.0, 249452.0, 249737.0, 250013.0, 250286.0, 250286.0, 250962.0, 251208.0, 251572.0, 251843.0, 252296.0, 252688.0, 252688.0, 253362.0, 253793.0, 254323.0, 254852.0, 254852.0, 256338.0, 257276.0])
#active = [int(confirmed[i] - deaths[i] - recovered[i]) for i in range(len(confirmed))]
active = [54,56,53,52,50,48,39,40,38,41,33,31,30,29,30,32,30,32,33,36,48,48,60,67,73,82,91,103,107,121,134,152,196,221,254,290,309,355,400,469,509,547,602,629,648,683,752,779,827,886,983,1025,1098,1211,1444,1609,1763,1964,2323,2631,3037,3734,4331,5241,5809,7202,8275,9233,10242,11107,11679,12552,13314,13809,14439,14910,15817,16184,16779,17303,17873,18544,19207,19647,20144,20595,20541,20799,20516,20104,19622,18992,18676,18486,18407,18135,17672,17408,17163,16717,16199,15876,15577,14932,14206,13616,13162,12802,12637,12799,12994,12950,12943,12999,12903,12612,12408,12076,11785,11363,10989,10426,9780,9252,8735,8130,7583,7127,6697,6411,6298,6106,6104,6057,5925,5650,5381,5085,4855,4684,4521,4333,4240,4111,3948,3751,3807,3731,3650,3716,3865,3863,3843,3849,3795,3799,3637,3823,3922,4056,4176,4509,4821,5119,5277,5406,5474,5687,5745,5872,5845,5865,6459,6497,6458,6319,6162,5656,5198,4848,4734,4504,4113,3767,3461,3378,3194,2953,2538,2319,2162,1790,1592,1497,1406,1302,1243,1158,1127,1076,942,853,747,688,662,609,603,647,644,681,631,615,625,577,532,466,445,389,368,338,338,321,294,297,299,306,295,249,250,245,233,211,210,195,191,189,154,157,141,144,124,116,109,100,89,78,76,68,74,84,84,91,93,84,66,69,69,67,66,74,67,68,64,70,66,60,58,53,55,60,73,72,78,69,62,63,63,61,59,57,56,65,66,76,71,63,60,57,60,55,60,57,65,61,68,71,76,80,80,80,84,87,83,86,79,86,96,92,100,114,116,128,131,134,128,128,125,114,113,129,121,151,157,181,195,203,210,222,227,225,242,232,223,233,243,259,270,238,230,234,242,247,238,216,238,257,251,258,248,239,282,279,265,254,253,247,247,241,237,208,197,192,172,179,153,150,139,120,116,124,120,110,109,104,101,100,86,81,80,84,89,85,101,112,108,112,110,117,118,122,102,101,97,107,113,114,106,108,118,132,128,127,128,128,137,132,145,148,160,179,202,216,244,253,246,251,250,264,261,267,281,288,291,288,297,288,293,315,316,318,295,274,271,300,307,314,339,329,338,373,364,383,401,398,398,393,382,374,388,395,394,397,382,393,427,443,450,459,486,475,469,496,490,498,512,529,524,536,531,548,561,559,555,544,555,545,530,528,517,483,449,437,412,391,373,373,370,374,372,361,346,338,325,315,322,318,317,313,313,305,308,315,319,313,305,295,295,291,275,269,268,251,242,250,255,301,335,379,433,511,677,861,1025,1179,1301,1422,1537,1660,1779,1889,2006,2091,2081,2108,2142,2025,2014,2011,2028,1986,1864,1732,1696,1617,1589,1480,1406,1336,1270,1174,1136,1078,957,868,828,826,870,935,967,1027,1094,1157,1272,1391,1523,1624,1787,1902,2005,2246,2436,2314,2670,2787,3121,3441,3801,4473,5003,5538,6110,6608,7144,8059,8206,9198,10379,11042,11742,13162,14283,15469,16643,18252,20203,20952,21804,21731,23463,24138,24173,25303,25976,26307,26311,27166,27132,26622,26775,26685,25980,25456,26908,28037,28854,29652,29732,29731,29937,30348,32490,31559,32780,31966,31357,30261,30742,31025,28924,27025,26177,25787,25027,25578,26547,24525,24631,24383,23893,22684,22374,23286,21546,20390,19775,18797,18126,18125,17923,15972,14826,14684,13359,12642,12474,12255,11291,10655,9875,8896,8356,8179,7772,6975,6408,6005,5540,5206,5039,4874,4381,4222,3991,3740,3537,3387,3227,3547,3006,2874,2738,2696,2776,2840,2877,2868,2999,3182,3282,3850,4200,4620,5395,5532,5945,6165,6482,6727,6837,6843,7086,7125,7591,8267,9195]
active = [0 for i in range(25)] + active[:-6]

susceptible = [N - confirmed[i] for i in range(len(confirmed))]
data = np.c_[susceptible,active,recovered,deaths]

beta = 0.4 #0.4 to 1
gamma = 1/8 # 1/9 to 1/7
mu = 0.0000525 #0.000005 to 0.0001

#proportionality factor for err in values of X, Theta
errX = 0.01
errTheta = 0.1

err_obs_X = 0.003
err_est_X = 0.003
err_obs_theta = 0.1
err_est_theta = 0.1
Q_err=10
R_err=10

H = np.diag([1,1,1,1])

def initX(start=0):
    #start: offset from the starting day, 21 jan 20
    #Initial conditions
    S = N - confirmed[start]
    I = active[start]
    R = recovered[start]
    D = deaths[start]
    return np.array([[S],
                    [I],
                    [R],
                    [D]])

def initTheta():
    return np.array([[beta],
                     [gamma],
                     [mu]])


def initP(X):
    [S,I,R,D] = [int(_) for _ in X]
    #print(S,I,R,D)
    return np.diag([S*err_obs_X,I*err_obs_X,R*err_obs_X,D*err_obs_X])


def f(X,theta):
    #X, theta are the vectors at time t
    [S,I,R,D] = [_[0] for _ in X]
    [beta,gamma,mu] = [_[0] for _ in theta]
    
    
    Sp = S - (beta*I*S)/N
    Ip = I + (beta*I*S)/N - (gamma+mu)*I
    Rp = R + gamma*I
    Dp = D + mu*I
    
    if Ip == 0:
        Ip = 1
    
    Xp = np.array([[Sp],
                   [Ip],
                   [Rp],
                   [Dp]])
    #Xp = predicted future X value for time = t+1
    return Xp



def getF(X,theta):
    [S,I,R,D] = [_[0] for _ in X]
    [beta,gamma,mu] = [_[0] for _ in theta]
    return np.array([[1-beta*I/N, -beta*S/N, 0, 0],
                     [beta*I/N,  1+beta*S/N, 0, 0],
                     [0,         gamma,      1, 0],
                     [0,             0,     mu, 1]])

def getQ(X, mode=0):
    if mode==0:
        [S,I,R,D] = [_[0] for _ in X]
        return np.diag([err_est_X*S,err_est_X*I,err_est_X*R,err_est_X*D])
    if mode==1:
        return np.diag([Q_err,Q_err,Q_err,Q_err])

def getR(X, mode=1):
    if mode==0:
        [S,I,R,D] = [_[0] for _ in X]
        return np.diag([S*err_obs_X,I*err_obs_X,R*err_obs_X,D*err_obs_X])
    if mode==1:
        return np.diag([R_err,R_err,R_err,R_err])

def predictP(P,F,Q):
    #returns predictedP
    #assumption: S,I,R,D dont have any covariance between each other
    Qfactor = 0
    
    main = np.diag(np.diag(F.dot(P).dot(F.T)))
    Q = Qfactor*main
    return np.diag(np.diag(F.dot(P).dot(F.T))) + Q

def addNoiseToXp(Xp,Q,factor=1):
    varList = Q.dot(np.array([[1],[1],[1],[1]]))
    stddevList = [math.sqrt(_[0]) for _ in varList]
    noiseList = [factor*np.random.normal(0,_,1)[0] for _ in stddevList]
    noiseList[3] = 0
    noiseVec = np.array([_ for _ in noiseList])
    return Xp + noiseVec

def addNoiseToTheta(theta,varList):
    stddevList = [math.sqrt(_) for _ in varList]
    noiseList = [[np.random.normal(0,_,1)[0]] for _ in stddevList]
    noiseVec = np.array(noiseList).reshape(3,1)
    return theta + noiseVec
    
#-----------------------------------------------------------------------------
def getK(Pp,H,R):
    return Pp.dot(H).dot(inv(H.dot(Pp).dot(H.T) + R))

def getZ(t):
    l = data[t]
    return np.array([[l[0]],
                     [l[1]],
                     [l[2]],
                     [l[3]]])

def updateX(Xp, K, H, z):
    #print(z)
    #print(Xp)
    #print(H.dot(Xp))
    #print(z - H.dot(Xp))
    return Xp + K.dot(z - H.dot(Xp))

def updateP(K,H,Pp):
    return (np.identity(4) - K.dot(H)).dot(Pp)

def updateTheta(z_new, z_old, theta):
    [S_old, I_old, R_old, D_old] = [_[0] for _ in z_old]
    [S_new, I_new, R_new, D_new] = [_[0] for _ in z_new]
    [beta_old, gamma_old, mu_old] = theta.tolist()
    obv_beta = - (S_new-S_old) * N/(S_old*I_old)
    obv_gamma = (R_new-R_old) / I_old
    obv_mu = (D_new-D_old) / I_old
    
    #return np.array([(beta_old+obv_beta)/2, (gamma_old+obv_gamma)/2, (mu_old+obv_mu)/2])
    return np.array([[obv_beta], [obv_gamma], [obv_mu]])

def plot(x,ax=0):
    if ax==0:
        plt.plot(x)
        plt.show()
    else:
        ax.plot(x)
        plt.show()





def predict(X,theta,P):
    #Q not needed, only returned for convenience for addNoiseToXp()
    Xp = f(X,theta)
    F = getF(X,theta)
    Q = getQ(X,mode=1)
    R = getR(X)
    Pp = predictP(P,F,Q)
    return Xp, Pp, R, Q

def get_measurements(t):
    #minimum t=1
    return getZ(t), getZ(t-1)
    
def update(Xp,Pp,R,H,z,z_old,theta):
    K = getK(Pp,H,R)
    X = updateX(Xp,K,H,z)
    P = updateP(K,H,Pp)
    theta = updateTheta(z,z_old,theta)
    return X, P, theta

def simulation(start,endpt,forecast=0,randfactor=0,forecastFactor=0):
    output = []; outputTheta = []
    
    X = initX(start)
    theta = initTheta()
    P = initP(X)
    
    output.append([_[0] for _ in X])
    for t in range(start+1,endpt):
        Xp, Pp, R, Q = predict(X,theta,P)
        Xp = addNoiseToXp(Xp,Q,randfactor)
        #print(Xp)

        z, z_old = get_measurements(t)
        X, P, theta = update(Xp,Pp,R,H,z,z_old,theta)
        output.append([_[0] for _ in X])
        outputTheta.append(theta)
    
    Xp_list = []; Pp_list = []
    for t in range(forecast):
        Xp, Pp, R, Q = predict(X,theta,P)
        Xp = addNoiseToXp(Xp,Q,forecastFactor)
        
        Xp_list.append(Xp)
        Pp_list.append(Pp)
        if t == 0:
            z, z_old = Xp, getZ(len(data)-1)
        else:
            z, z_old = Xp_list[-1], Xp_list[-2]
        X, P, theta = update(Xp,Pp,R,H,z,z_old,theta)
        #theta = addNoiseToTheta(theta,[0.001,0.05,0.00002])
        
        output.append([_[0] for _ in X])
        outputTheta.append(theta)
        
        #print(z)
        #print(z_old, t)
    return output, outputTheta


lenForecast = 40
output, outputTheta = simulation(25,len(data),forecast=lenForecast,randfactor=0)

s_output = [_[0] for _ in output]
i_output = [int(_[1]) for _ in output]
r_output = [_[2] for _ in output]
d_output = [_[3] for _ in output]
bl = [_[0] for _ in outputTheta]
gammaList = [_[1] for _ in outputTheta]
muList = [_[2] for _ in outputTheta]


outputLists = []; outputThetaLists = []
for i in range(12):
    o, ot = simulation(25,len(data),forecast=lenForecast,randfactor=0,forecastFactor=5)
    outputLists.append(o[-lenForecast:])
    outputThetaLists.append(ot[-lenForecast:])

i_lists = [[int(_[1]) for _ in output] for output in outputLists]
#print(i_lists)
def stochastic_graphs(lists, z, forecastLength):
    #lists: list of list, each list contains 1 random simulation of a variable over time
    #sd: desired sd of graphs to be returned
    n_sims = len(lists)
    output_low, output_high = [],[]
    mean_list = []
    sd_list = []
    for i in range(forecastLength):
        mean_list.append(np.mean([x[i] for x in lists]))
        sd_list.append(np.std([x[i] for x in lists]))
    output_low = [(mean_list[i] - z*sd_list[i]/math.sqrt(n_sims)) for i in range(forecastLength)]
    output_high = [(mean_list[i] + z*sd_list[i]/math.sqrt(n_sims)) for i in range(forecastLength)]

    return output_low, output_high, mean_list

def plot_ci(ax,normal,lo,hi,_color,_label):
    #plot expected graph with CI
    ax.plot(normal,label=_label,color=_color)
    ax.fill_between(np.array([i for i in range(len(normal))]),lo,hi,alpha=0.2,color=_color)

def combine(expected,mean):
    #combine expected graph and mean of outputs
    return [(0.05*expected[i] + 0.95*mean[i]) for i in range(len(expected))]
    
    
lo, hi, mean = stochastic_graphs(i_lists,1.96,lenForecast)
#print(lo,hi)

#print(len(i_output))
i_lo_end = [int(_) for _ in lo]
i_hi_end = [int(_) for _ in hi]
i_mean_end = [int(_) for _ in mean]
i_lo = i_output[:-lenForecast] + i_lo_end
i_hi = i_output[:-lenForecast] + i_hi_end

combined_end = combine(i_output[-lenForecast:],mean)
combined = i_output[:-lenForecast] + combined_end


fig,ax = plt.subplots()
plot_ci(ax,combined[-50:],i_lo[-50:],i_hi[-50:],'blue','i')
ax.axvline(x=50-lenForecast,linestyle='--')
plt.show()

def rmse(arr1,arr2):
    if len(arr1) != len(arr2):
        return 'diff length!'
    total = 0
    for i in range(len(arr1)):
        total += (arr1[i]-arr2[i])**2
    return math.sqrt(total/len(arr1))

def nrmse(lists, expected, pos):
    exp = expected[pos]
    values = [l[pos] for l in lists]
    n = len(values)
    
    total = 0
    for y in values:
        total += (exp-y)**2
    total = math.sqrt(total/n)
    q3, q1 = getInterquartile(values)
    return total/(exp)
    
def getInterquartile(values):
    n = len(values)
    i3 = int(n*0.25)
    i1 = int(n*0.75)
    sorted_vals = sorted(values)
    return sorted_vals[i3], sorted_vals[i1]

nrmseList = []
for i in range(lenForecast):
    nrmseList.append(nrmse(i_lists,i_mean_end,i))
plot(nrmseList)
