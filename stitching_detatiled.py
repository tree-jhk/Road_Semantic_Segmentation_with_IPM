# -*- coding: utf-8 -*-
"""
Stitching sample (advanced)
===========================
Show how to use Stitcher API from python.
"""
# Python 2/3 compatibility

# ************************************************************************************************************************************************************************************************************************************************

from __future__ import print_function
# 파이썬2랑 파이썬3 버전 차이나서 print같은 놈이 다르게 인식이 됨. 그래서 그냥 2,3 버전 상관없이 진행할 수 있도록 __future__를 임포트시켜준다.
# 자세한 것 참조: https://teknology.tistory.com/5
# 따라서 __future__를 한번 임포트시키면 구버전에서도 미래 버전의 기능을 사용할 수 있다.
import argparse
from collections import OrderedDict
# 자세한 것 참조: https://www.daleseo.com/python-collections-ordered-dict/
# 자세한 것 참조: https://bio-info.tistory.com/52
# 참고: 일반적인 dict 자료형은 하나의 dict형 변수에 여러 개의 key와 value 쌍을 추가했을 때, key와 value가 순서가 랜덤으로 받아져서 key와 value가 처음에 넣었던 것과 다르게 된다. 그래서 OrderedDict를 import해서 key와 value가 삽입된 순서를 기억하게 한다.
import cv2 as cv
import numpy as np

# ************************************************************************************************************************************************************************************************************************************************

EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO

# ************************************************************************************************************************************************************************************************************************************************

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster

# ************************************************************************************************************************************************************************************************************************************************

FEATURES_FIND_CHOICES = OrderedDict()

FEATURES_FIND_CHOICES['orb'] = cv.ORB.create
try:
    FEATURES_FIND_CHOICES['sift'] = cv.SIFT_create
except AttributeError:
    print("SIFT not available")
    # AttributeError: 속성(attribute) 이름이 잘못됐거나 없는 속성을 가져오려 하면 속성 오류(AttributeError)가 발생합니다.
try:
    FEATURES_FIND_CHOICES['brisk'] = cv.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['akaze'] = cv.AKAZE_create
except AttributeError:
    print("AKAZE not available")
    
# ************************************************************************************************************************************************************************************************************************************************
    
SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
# print("뭘까",(SEAM_FIND_CHOICES.values()))
print("뭘까",dir(SEAM_FIND_CHOICES.values))
SEAM_FIND_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
SEAM_FIND_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)
# seam: 이음새, 봉제선이라는 뜻.

# ************************************************************************************************************************************************************************************************************************************************

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator
ESTIMATOR_CHOICES['affine'] = cv.detail_AffineBasedEstimator

# ************************************************************************************************************************************************************************************************************************************************

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

# ************************************************************************************************************************************************************************************************************************************************

WAVE_CORRECT_CHOICES = OrderedDict()
WAVE_CORRECT_CHOICES['horiz'] = cv.detail.WAVE_CORRECT_HORIZ
WAVE_CORRECT_CHOICES['no'] = None
WAVE_CORRECT_CHOICES['vert'] = cv.detail.WAVE_CORRECT_VERT

# ************************************************************************************************************************************************************************************************************************************************

BLEND_CHOICES = ('multiband', 'feather', 'no',)

# ************************************************************************************************************************************************************************************************************************************************

print(type(EXPOS_COMP_CHOICES))
print(type(BA_COST_CHOICES))
print(type(FEATURES_FIND_CHOICES))
print(type(SEAM_FIND_CHOICES))
print(type(ESTIMATOR_CHOICES))
print(type(WARP_CHOICES))
print(type(WAVE_CORRECT_CHOICES))
print(type(BLEND_CHOICES), "\n")

# 출력 결과:
# <class 'collections.OrderedDict'> / collections.OrderedDict: 정렬된 dic형 자료형이라 생각하면 된다.
# <class 'collections.OrderedDict'>
# <class 'collections.OrderedDict'>
# <class 'collections.OrderedDict'>
# <class 'collections.OrderedDict'>
# <class 'tuple'>
# <class 'collections.OrderedDict'>
# <class 'tuple'>

print("EXPOS_COMP_CHOICES 의 dictionary구성: ", (EXPOS_COMP_CHOICES),"\n")
print("BA_COST_CHOICES 의 dictionary구성: ", (BA_COST_CHOICES),"\n")
print("FEATURES_FIND_CHOICES 의 dictionary구성: ", (FEATURES_FIND_CHOICES),"\n")
print("SEAM_FIND_CHOICES 의 dictionary구성: ", (SEAM_FIND_CHOICES),"\n")
print("ESTIMATOR_CHOICES 의 dictionary구성: ", (ESTIMATOR_CHOICES),"\n")
print("WARP_CHOICES 의 dictionary구성: ", (WARP_CHOICES),"\n")
print("WAVE_CORRECT_CHOICES 의 dictionary구성: ", (WAVE_CORRECT_CHOICES),"\n")
print("BLEND_CHOICES 의 dictionary구성: ", (BLEND_CHOICES),"\n")

parser = argparse.ArgumentParser(
    prog="stitching_detailed.py", description="Rotation model images stitcher"
)
parser.add_argument(
    'img_names', nargs='+',
    help="Files to stitch", type=str
    # nargs='+': 값 개수 지정 :1개 이상의 값을 전부 읽어들인다. 정규표현식의 것과 매우 비슷하다.
    # type: 기본적으로 parse_args()가 주어진 인자들을 파싱할 때는 모든 문자를 숫자 등이 아닌 문자열 취급한다. 따라서 데이터 타입을 지정하고 싶으면 add_argument()에서 type=을 지정해 주어야 한다. default는 말한 대로 str이다.
    # --help 또는 -h: 기본으로 내장되어 있는 옵션이다. 이 인자를 넣고 python으로 실행하면 인자 사용법에 대한 도움말이 출력된다.
)
parser.add_argument(
    '--try_cuda',
    action='store',
    default=False,
    help="Try to use CUDA. The default value is no. All default values are for CPU mode.",
    type=bool, dest='try_cuda'
)
parser.add_argument(
    '--work_megapix', action='store', default=0.6,
    help="Resolution for image registration step. The default is 0.6 Mpx",
    type=float, dest='work_megapix'
)
# work_scale이랑 관련 있음.
parser.add_argument(
    '--features', action='store', default=list(FEATURES_FIND_CHOICES.keys())[0],
    help="Type of features used for images matching. The default is '%s'." % list(FEATURES_FIND_CHOICES.keys())[0],
    choices=FEATURES_FIND_CHOICES.keys(),
    type=str, dest='features'
)
parser.add_argument(
    '--matcher', action='store', default='homography',
    help="Matcher used for pairwise image matching. The default is 'homography'.",
    choices=('homography', 'affine'),
    type=str, dest='matcher'
)
parser.add_argument(
    '--estimator', action='store', default=list(ESTIMATOR_CHOICES.keys())[0],
    help="Type of estimator used for transformation estimation. The default is '%s'." % list(ESTIMATOR_CHOICES.keys())[0],
    choices=ESTIMATOR_CHOICES.keys(),
    type=str, dest='estimator'
)
parser.add_argument(
    '--match_conf', action='store',
    help="Confidence for feature matching step. The default is 0.3 for ORB and 0.65 for other feature types.",
    type=float, dest='match_conf'
)
parser.add_argument(
    '--conf_thresh', action='store', default=1.0,
    help="Threshold for two images are from the same panorama confidence.The default is 1.0.",
    type=float, dest='conf_thresh'
)
parser.add_argument(
    '--ba', action='store', default=list(BA_COST_CHOICES.keys())[0],
    help="Bundle adjustment cost function. The default is '%s'." % list(BA_COST_CHOICES.keys())[0],
    choices=BA_COST_CHOICES.keys(),
    type=str, dest='ba'
)
parser.add_argument(
    '--ba_refine_mask', action='store', default='xxxxx',
    help="Set refinement mask for bundle adjustment. It looks like 'x_xxx', "
         "where 'x' means refine respective parameter and '_' means don't refine, "
         "and has the following format:<fx><skew><ppx><aspect><ppy>. "
         "The default mask is 'xxxxx'. "
         "If bundle adjustment doesn't support estimation of selected parameter then "
         "the respective flag is ignored.",
    type=str, dest='ba_refine_mask'
)
parser.add_argument(
    '--wave_correct', action='store', default=list(WAVE_CORRECT_CHOICES.keys())[0],
    help="Perform wave effect correction. The default is '%s'" % list(WAVE_CORRECT_CHOICES.keys())[0],
    choices=WAVE_CORRECT_CHOICES.keys(),
    type=str, dest='wave_correct'
)
# wave_correct: 0: 좀 더 horiztal 방향으로, 1: 안바꾸고, 2: vertical 방향으로
# 이미 수평이 완벽한 경우라면 [0]이나 [1]이나 차이가 없고, 2로 하면 90도 회전한 것 같음.
parser.add_argument(
    '--save_graph', action='store', default=None,
    help="Save matches graph represented in DOT language to <file_name> file.",
    type=str, dest='save_graph'
)
parser.add_argument(
    '--warp', action='store', default=WARP_CHOICES[0],
    help="Warp surface type. The default is '%s'." % WARP_CHOICES[0],
    choices=WARP_CHOICES,
    type=str, dest='warp'
)
parser.add_argument(
    '--seam_megapix', action='store', default=0.1,
    help="Resolution for seam estimation step. The default is 0.1 Mpx.",
    type=float, dest='seam_megapix'
)
parser.add_argument(
    '--seam', action='store', default=list(SEAM_FIND_CHOICES.keys())[0],
    help="Seam estimation method. The default is '%s'." % list(SEAM_FIND_CHOICES.keys())[0],
    choices=SEAM_FIND_CHOICES.keys(),
    type=str, dest='seam'
)
parser.add_argument(
    '--compose_megapix', action='store', default=-1,
    help="Resolution for compositing step. Use -1 for original resolution. The default is -1",
    type=float, dest='compose_megapix'
)
parser.add_argument(
    '--expos_comp', action='store', default=list(EXPOS_COMP_CHOICES.keys())[0],
    help="Exposure compensation method. The default is '%s'." % list(EXPOS_COMP_CHOICES.keys())[0],
    choices=EXPOS_COMP_CHOICES.keys(),
    type=str, dest='expos_comp'
)
parser.add_argument(
    '--expos_comp_nr_feeds', action='store', default=1,
    help="Number of exposure compensation feed.",
    type=np.int32, dest='expos_comp_nr_feeds'
)
# parser.add_argument(
#     '--expos_comp_nr_filtering', action='store', default=2,
#     help="Number of filtering iterations of the exposure compensation gains.",
#     type=float, dest='expos_comp_nr_filtering'
# )
parser.add_argument(
    '--expos_comp_block_size', action='store', default=32,
    help="BLock size in pixels used by the exposure compensator. The default is 32.",
    type=np.int32, dest='expos_comp_block_size'
)
parser.add_argument(
    '--blend', action='store', default=BLEND_CHOICES[0],
    help="Blending method. The default is '%s'." % BLEND_CHOICES[0],
    choices=BLEND_CHOICES,
    type=str, dest='blend'
)
parser.add_argument(
    '--blend_strength', action='store', default=5,
    help="Blending strength from [0,100] range. The default is 5",
    type=np.int32, dest='blend_strength'
)
parser.add_argument(
    '--output', action='store', default='result.jpg',
    help="The default is 'result.jpg'",
    type=str, dest='output'
)
parser.add_argument(
    '--timelapse', action='store', default=None,
    help="Output warped images separately as frames of a time lapse movie, "
         "with 'fixed_' prepended to input file names.",
    type=str, dest='timelapse'
)
parser.add_argument(
    '--rangewidth', action='store', default=-1,
    help="uses range_width to limit number of images to match with.",
    type=int, dest='rangewidth'
)

# ************************************************************************************************************************************************************************************************************************************************

__doc__ += '\n' + parser.format_help()

# ************************************************************************************************************************************************************************************************************************************************

def get_matcher(args):
    try_cuda = args.try_cuda
    matcher_type = args.matcher
    if args.match_conf is None:
        if args.features == 'orb':
            match_conf = 0.3
        else:
            match_conf = 0.65
    else:
        match_conf = args.match_conf
    range_width = args.rangewidth
    if matcher_type == "affine":
        matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
    elif range_width == -1:
        matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)
    else:
        matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda, match_conf)
    return matcher

# ************************************************************************************************************************************************************************************************************************************************

def get_compensator(args):
    expos_comp_type = EXPOS_COMP_CHOICES[args.expos_comp]
    expos_comp_nr_feeds = args.expos_comp_nr_feeds
    expos_comp_block_size = args.expos_comp_block_size
    # expos_comp_nr_filtering = args.expos_comp_nr_filtering
    if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
        compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator

# ************************************************************************************************************************************************************************************************************************************************

def main():
    
    args = parser.parse_args()     
    img_names = args.img_names
    print(img_names)
    work_megapix = args.work_megapix
    seam_megapix = args.seam_megapix
    compose_megapix = args.compose_megapix
    conf_thresh = args.conf_thresh
    ba_refine_mask = args.ba_refine_mask
    wave_correct = WAVE_CORRECT_CHOICES[args.wave_correct]
    if args.save_graph is None:
        save_graph = False
    else:
        save_graph = True
    warp_type = args.warp
    blend_type = args.blend
    blend_strength = args.blend_strength
    result_name = args.output
    if args.timelapse is not None:
        timelapse = True
        if args.timelapse == "as_is":
            timelapse_type = cv.detail.Timelapser_AS_IS
        elif args.timelapse == "crop":
            timelapse_type = cv.detail.Timelapser_CROP
        else:
            print("Bad timelapse method")
            exit()
    else:
        timelapse = False
    finder = FEATURES_FIND_CHOICES[args.features]()
    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    for name in img_names:
        full_img = cv.imread(cv.samples.findFile(name))
        if full_img is None:
            print("Cannot read image ", name)
            exit()
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        img_feat = cv.detail.computeImageFeatures2(finder, img)
        features.append(img_feat)
        img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        images.append(img)
    matcher = get_matcher(args)
    p = matcher.apply2(features)
    matcher.collectGarbage()
    if save_graph:
        with open(args.save_graph, 'w') as fh:
            fh.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))
    indices = cv.detail.leaveBiggestComponent(features, p, conf_thresh)
    img_subset = []
    img_names_subset = []
    full_img_sizes_subset = []
    ##==========  indices[i, 0]을 indices[i] 로 바꿈. ================###
    for i in range(len(indices)):
        img_names_subset.append(img_names[indices[i]])
        img_subset.append(images[indices[i]])
        full_img_sizes_subset.append(full_img_sizes[indices[i]])
    #####################################################################
    images = img_subset
    img_names = img_names_subset
    full_img_sizes = full_img_sizes_subset
    num_images = len(img_names)
    if num_images < 2:
        print("Need more images")
        exit()
    estimator = ESTIMATOR_CHOICES[args.estimator]()
    b, cameras = estimator.apply(features, p, None)
    # 이거 회전행렬 같은데?
    print("이거 냄새가 솔솔 나는디????",type(estimator))
    print("이거 냄새가 솔솔 나는디????", b)
    print("이거 냄새가 솔솔 나는디????", cameras)
    if not b:
        print("Homography estimation failed.")
        exit()
    # cnt = 0
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
        print(cam.R)
        # ★★★중요회전행렬!rotation matrix
        # cnt+=1
        # print(cnt)
        # 역시나 이미지 개수만큼 생긴다.
    adjuster = BA_COST_CHOICES[args.ba]()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        exit()
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    if wave_correct is not None:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
            # print(rmats)
            # print(type(rmats))
            # 회전행렬을 copy해서 rmats에 list형으로 저장함.
        rmats = cv.detail.waveCorrect(rmats, wave_correct)
        # wave_correct: Tries to make panorama more horizontal (or vertical).
        # default값을 따라서, wave_correct가 horziontal방향으로 설정됨.
        # 행렬 자체가 바뀔 수는 있는데, 육안상으로는 바뀐 것으로 안 보일 수가 있음.
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
            # print(cameras)
            # print(cam.R)
            # print(type(cam.R))
            # 위에 copy했던 회전행렬에 waveCorrect(?)를 먹이고 나서 ndarry형으로 저장한다.
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        # 여기서 num_images는    명령어:   python3 stitching_detatiled.py --estimator homography "파일/경로/0번째_이미지.jpg" "파일/경로/1번째_이미지.jpg" ...  "파일/경로/(n - 1)번째_이미지.jpg"
        # 에서 이미지의 크기를 말한다.
        # num_images = len(img_names)
        
        um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        # UMat 클래스: UMat은 OpenCV가 OpenCL 기반의 코드로 이미지를 처리하게 한다.
        # 보통은 UMat말고 Mat을 쓰는데, UMat으로 쓰면 성능이 더 좋다고 함.
        # 결론: UMat 클래스는 사실 일종의 데이터타입이고, Mat 대신 쓴 거임.
        # Mat은 opencv에서 가장 기본이 되는 데이터 타입으로, 행렬 구조체이다.
        masks.append(um)
        # print("\n","*************행렬인가?? 뭐냐 얘는??******",type(um),"*************행렬인가?? 뭐냐 얘는??******","\n")
        
        # print("\n","*************행렬인가?? 뭐냐 얘는??******",type(images[i].shape[0]),"*************행렬인가?? 뭐냐 얘는??******","\n")
        # print("\n","*************행렬인가?? 뭐냐 얘는??******",type(images[i].shape[1]),"*************행렬인가?? 뭐냐 얘는??******","\n")
        # print("\n","*************행렬인가?? 뭐냐 얘는??******",type(images[i].shape[2]),"*************행렬인가?? 뭐냐 얘는??******","\n")
        
        # print("\n","*************행렬인가?? 뭐냐 얘는??******", (images[i].shape[0]),"*************행렬인가?? 뭐냐 얘는??******","\n")
        # print("\n","*************행렬인가?? 뭐냐 얘는??******", (images[i].shape[1]),"*************행렬인가?? 뭐냐 얘는??******","\n")
        # print("\n","*************행렬인가?? 뭐냐 얘는??******", (images[i].shape[2]),"*************행렬인가?? 뭐냐 얘는??******","\n")
    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        # 회전행렬 객체를 K에 삽입
        swa = seam_work_aspect
        K[0, 0] *= swa
        # K[0, 0] = K[0, 0] * swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())
    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)
    compensator = get_compensator(args)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)
    seam_finder = SEAM_FIND_CHOICES[args.seam]
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    timelapser = None
    # https://github.com/opencv/opencv/blob/4.x/samples/cpp/stitching_detailed.cpp#L725 ?
    for idx, name in enumerate(img_names):
        full_img = cv.imread(name)
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(img_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                      int(round(full_img_sizes[i][1] * compose_scale)))
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])
        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img
        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv.dilate(masks_warped[idx], None)
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)
        if blender is None and not timelapse:
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        elif timelapser is None and timelapse:
            timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
            timelapser.initialize(corners, sizes)
        if timelapse:
            ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
            timelapser.process(image_warped_s, ma_tones, corners[idx])
            pos_s = img_names[idx].rfind("/")
            if pos_s == -1:
                fixed_file_name = "fixed_" + img_names[idx]
            else:
                fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
            cv.imwrite(fixed_file_name, timelapser.getDst())
        else:
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
    if not timelapse:
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        # print(result.shape)
        # result: rgb 결과 이미지
        # print(result_mask.shape)
        # result_mask: 결과 이미지를 출력할 틀 / (2297, 3476)
        
        # print("\n","출력 결과: type(work_scale): ", type(work_scale), "\n","출력 결과: type(work_megapix): ", type(work_megapix))
        # print("\n","출력 결과: type(seam_finder): ", type(seam_finder),"\n","출력 결과: type(seam_mask): ", type(seam_mask))
        # print("\n","출력 결과: type(seam_megapix): ", type(seam_megapix), "\n","출력 결과: type(seam_scale): ", type(seam_scale), "\n","출력 결과: type(seam_work_aspect): ", type(seam_work_aspect),"\n")
#  출력 결과: type(work_scale):  <class 'numpy.float64'> 
#  출력 결과: type(work_megapix):  <class 'float'>

#  출력 결과: type(seam_finder):  <class 'cv2.detail_DpSeamFinder'> 
#  출력 결과: type(seam_mask):  <class 'cv2.UMat'>

#  출력 결과: type(seam_megapix):  <class 'float'> 
#  출력 결과: type(seam_scale):  <class 'numpy.float64'> 
#  출력 결과: type(seam_work_aspect):  <class 'numpy.float64'>
        
        # print("\n","출력 결과: work_scale.shape: ", work_scale.shape, "\n","출력 결과: work_megapix.shape: ", work_megapix)
        # print("\n","출력 결과: seam_finder.shape: ", dir(seam_finder),"\n","출력 결과: seam_mask.shape: ", dir(seam_mask))
        # print("\n","출력 결과: seam_megapix.shape: ", seam_megapix, "\n","출력 결과: seam_scale.shape: ", seam_scale.shape, "\n","출력 결과: seam_work_aspect.shape: ", seam_work_aspect.shape,"\n")
#  출력 결과: work_scale.shape:  () 
#  출력 결과: work_megapix.shape:  0.6

#  출력 결과: seam_finder.shape:  ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'createDefault', 'find', 'setCostFunction'] 
#  출력 결과: seam_mask.shape:  ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'context', 'get', 'handle', 'isContinuous', 'isSubmatrix', 'offset', 'queue']

#  출력 결과: seam_megapix.shape:  0.1 
#  출력 결과: seam_scale.shape:  () 
#  출력 결과: seam_work_aspect.shape:  ()
        
        cv.imwrite(result_name, result)
        zoom_x = 600.0 / result.shape[1]
        dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # alpha: 밝기, NORM_MINMAX: normalize의 방법 중 하나, output array of type ref CV_8U that has the same size and the same number of channels as the input arrays.
        dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        # print(result.shape)
        # (2297, 3476, 3)
        cv.imshow(result_name, dst)     
        cv.waitKey()
    print("Done")

# ************************************************************************************************************************************************************************************************************************************************

if __name__ == '__main__':
    # print(__doc__) : 얘는 그냥 parser들의 help를 모두 읽음. 도큐먼트 다 출력해줌. 일단 주석처리함.
    
    
    # f1="/home/jhk/stitching/test/my_room/0001.jpg"
    # f2="/home/jhk/stitching/test/my_room/0002.jpg"
    # c1 = cv.imread(f1)
    # c2 = cv.imread(f2)
    # d1=cv.resize(c1,(600,400))
    # # d2=cv.resize(c2,(600,400))
    # cv.imshow('f1',d1)
    # cv.imshow('f2',d2)
    main()
    cv.destroyAllWindows()