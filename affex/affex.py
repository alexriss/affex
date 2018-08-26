"""
Image manipulation:
  - Stack frames from a movie to create a mean image (including frame alignment). This can be use for long exposures.
  - Stabilize video and create trail effects (custom functions are possible).

some code based on:
  - https://github.com/maitek/image_stacking
  - https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
  
  
  
Alex Riss, 2018, GPL

"""

import cv2
import argparse
import numpy as np
import sys
#import imageio  # will be imported on demand below

from matplotlib import pylab as plt


class StackImages:
    """
    Class to stack images.
    The class is first initialized, then the _align_image function is called iteratively (with the current frame to align to as input). It returns the aligned image.
    All images will be aligned to the first image.
    Parameters:
      method      : method used to align frames (either ORB or ECC).
      warp_mode   : warp mode to use for alignment with the ECC method. One of 'translation', 'euclidean', 'affine', 'homography'
    """
    def __init__(self, method='ECC', warp_mode='affine'):
        self.method = method
        self.prev_image = None
        self.first_image = None
        
        if warp_mode=='translation':
            self.warp_mode = cv2.MOTION_TRANSLATION
        elif warp_mode=='euclidean':
            self.warp_mode = cv2.MOTION_EUCLIDEAN
        elif warp_mode=='affine':
            self.warp_mode = cv2.MOTION_AFFINE
        elif warp_mode=='homography' or self.method == 'ORB':
            self.warp_mode = cv2.MOTION_HOMOGRAPHY
        else:
            raise NotImplementedError('Parameter warp_mode ("%s") should be one of the following: translation, euclidean, affine, homography.' %warp_mode)
        if self.method == 'ORB':
            self.warp_mode = cv2.MOTION_HOMOGRAPHY            

        if self.method=="ECC":
            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                self.M = np.eye(3, 3, dtype=np.float32)
            else:
                self.M = np.eye(2, 3, dtype=np.float32)            
        elif self.method=="ORB":
            self.orb = cv2.ORB_create()
            #cv2.ocl.setUseOpenCL(False) # disable OpenCL to because of bug in ORB in OpenCV 3.1
            self.first_kp = None
            self.first_des = None
        else:
            raise NotImplementedError('Parameter method should be ECC or ORB.')
        
            
    def _align_image(self, image):
        image_bw = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert to gray scale floating point image
        if self.method=='ORB':
            kp = self.orb.detect(image_bw, None)
            kp, des = self.orb.compute(image_bw, kp)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # create BFMatcher object
        if self.first_image is None:
            self.first_image = image_bw
            if self.method == 'ORB':
                self.first_kp = kp
                self.first_des = des
        else:
            if self.method == 'ECC':
                #s, self.M = cv2.findTransformECC(image_bw, self.first_image, self.M, cv2.MOTION_HOMOGRAPHY)  # Estimate perspective transform
                s, self.M = cv2.findTransformECC(image_bw, self.first_image, self.M, self.warp_mode)  # Estimate affine transformation
            elif self.method == 'ORB':
                matches = matcher.match(self.first_des, des)
                matches = sorted(matches, key=lambda x: x.distance)
                src_pts = np.float32([self.first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                self.M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)  # Estimate perspective transformation
            h, w, _ = image.shape
            if self.prev_image is None:
                if self.warp_mode==cv2.MOTION_HOMOGRAPHY:
                    image = cv2.warpPerspective(image, self.M, (w, h))
                else:
                    image = cv2.warpAffine(image, self.M, (w, h))
            else:
                if self.warp_mode==cv2.MOTION_HOMOGRAPHY:
                    image = cv2.warpPerspective(image, self.M, (w, h),  self.prev_image, borderMode=cv2.BORDER_TRANSPARENT)  # use stacked_image as background, so we dont get black borders
                else:
                    image = cv2.warpAffine(image, self.M, (w, h),  self.prev_image, borderMode=cv2.BORDER_TRANSPARENT)  # use stacked_image as background, so we dont get black borders
        
        self.prev_image = image
        return image
        
        
def _get_first_last_frame(totalFrames, first, last):
    """Returns the first and last frame numbers given the total number of frames and fractional first and last variables."""
    firstFrame, lastFrame = 0, totalFrames-1
    if first>last:
        first, last = last, first
    first, last = max(first, 0), max(last, 0)
    first, last = min(first, 1), min(last, 1)
    if first>=0 and last>0:
        firstFrame, lastFrame = round(first*totalFrames), round(last*(totalFrames-1))
    return firstFrame, lastFrame

    
def stackMovie(video, first=0, last=0, frame_step=1, align=True, align_method='ECC', align_warp_mode='affine', process_info = {}):
    """Aligns and averages frames within a video files.
    Arguments:
        video        : filename or opencv instance of the video to load
        first        : fraction of the video where to start the stacking (between 0 and 1)
        last         : fraction of the video where to end the stacking (between 0 and 1)
        frame_step   : stack every n frames (default: 1, i.e. stack every frame)
        align        : if True, all the frames will be aligned with the first frame
        method       : method for feature mapping: ECC or ORB (default: ECC; it is slower but more accurate)
        process_info : dictionary (optional). The 'out_progress' key holds a float that gets updated with the percentual progress (0 to 1). If 'in_stop' key is True, then the function will stop. 'out_ended' will be set to True at the end of execution.
    Returns:
        (stacked_image, first_image, last_image): the stacked, first and last images of the video in openCV image format"""
        
    if align: aligner = StackImages(method=align_method, warp_mode=align_warp_mode)
    if isinstance(video, cv2.VideoCapture):
        cap = video
    else:
        cap = cv2.VideoCapture(video)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    firstFrame, lastFrame = _get_first_last_frame(totalFrames, first, last)
    
    update_progress = False
    if 'out_progress' in process_info or 'in_stop' in process_info:
        update_progress = True
    use_temp_images = False
    if 'in_use_temp_images' in process_info:
        if process_info['in_use_temp_images']:
            use_temp_images=True

    first_image = None
    stacked_image = None
    i = 0
    print('Total of frames: %s, First frame: %s, Last frame: %s' % (totalFrames, firstFrame, lastFrame))
    numFrames = len(np.arange(firstFrame, lastFrame+1, frame_step))
    progress_every = int(np.ceil(numFrames/10))
    progress_every = min(progress_every, 20)
    for frame_number in np.arange(firstFrame, lastFrame+1, frame_step):
        if frame_number >= firstFrame and frame_number <= lastFrame:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
            ret, image = cap.read()
                
            if not image is None:
                if stacked_image is None:
                    stacked_image = np.zeros_like(image).astype(np.float32)
                    first_image = image
                if align: image = aligner._align_image(image)
                stacked_image += image/numFrames

            if (lastFrame-firstFrame)>0:
                if i % progress_every == 0:  # update progress every 10th frame
                    _print_progressbar(frame_number-firstFrame, lastFrame-firstFrame)
                    if use_temp_images:
                        process_info['out_temp_images'] = (np.array(np.round(stacked_image*numFrames/(i+1)),dtype=np.uint8), first_image, image)
                    
            i+=1
            if update_progress:
                process_info['out_progress'] = (frame_number-firstFrame)/(lastFrame-firstFrame)
                if process_info['in_stop']:
                    stacked_image = stacked_image*numFrames/i
                    print('\nStop signal received.')
                    break
                    
    _print_progressbar(1, 1)
    print('\nNumber of frames aligned and stacked: %s\n' % i)
    stacked_image = np.array(np.round(stacked_image),dtype=np.uint8)
    
    process_info['out_ended'] = True
    return (stacked_image, first_image, image)     


def createVideoTrail(video, fname_video_out, output_format='MJPG', first=0, last=0,
        output_size=None, output_fps=None, output_loop=0, output_optimize=False,
        num_frames_bwd=3, num_frames_fwd=0, func_bwd=None, func_fwd=None, align=True, align_method='ORB', align_warp_mode='affine', process_info = {}):
    """Creates Trails in a video. Writes the created video file to the argument fname_video_out.
    
    Arguments:
        video             : filename or opencv instance of the video to load
        fname_video_out   : filename to save the video
        output_format     : string specifying the output video format
        first             : fraction of the video where to start processing (between 0 and 1)
        last              : fraction of the video where to end processing (between 0 and 1)
        output_size       : output size specifying (width, height) in pixels (default: None, i.e. unchanged). Alternatively, a single scaling factor can be specified.
        output_fps        : for gifs: frames per second (default: None, i.e unchanged)
        output_loop       : for gifs: number of loops (default: 0, i.e. infinite)
        output_optimize   : for gifs: optimize gif (default: False)
        num_frames_bwd    : max number of frames before the current frame to use for the computation of trails
        num_frames_fwd    : max number of frames after the current frame to use for the computation of trails
        func_bwd          : weighting function to use to mix in the frames before the current one, linear function will be used if none provided
        func_fwd          : weighting function to use to mix in the frames after the current one, linear function will be used if none provided
        align             : If True, all the frames will be aligned with the first frame
        align_method      : method to sue for alignment, either ORB or ECC
        align_warp_mode   : warp mode to use for ECC alignment. one of 'translation', 'euclidian', 'affine', 'homography'
        process_info      : dictionary (optional). Keys:
                            'out_progress': gets updated with the percentual progress (0 to 1).
                            'in_stop': If True, then the function will stop.
                            'out_ended': will be set to True at the end of execution.
                            'in_use_temp_images': If True, then temporary images will be generated.
                            'out_temp_images': the current stacked imaged, the first image and the last image, updated during the loop
    """
    
    if fname_video_out[-4:].lower()=='.gif':
        output_format = 'gif'
    if output_format.lower()=='gif':
        output_format = 'gif'
        import imageio  # I will do it here, as it is not needed otherwise
    
    if align:
        aligner = StackImages(method=align_method, warp_mode=align_warp_mode)
        
    update_progress = False
    if 'out_progress' in process_info or 'in_stop' in process_info:
        update_progress = True
    use_temp_images = False
    if 'in_use_temp_images' in process_info:
        if process_info['in_use_temp_images']:
            use_temp_images=True

    if not output_size is None:
        output_size_scale = False
        if hasattr(output_size, '__len__'):
            if len(output_size) != 2:
                output_size = None
        else:
            output_size_scale = True
    
    if isinstance(video, cv2.VideoCapture):
        cap = video
    else:
        cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if output_fps is None:
        output_fps = fps
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    firstFrame, lastFrame = _get_first_last_frame(totalFrames, first, last)

    if func_bwd is None:
        func_bwd = lambda x: (1+num_frames_bwd - x)/(2*(1+num_frames_bwd))
    if func_fwd is None:
        func_fwd = lambda x: (1+num_frames_fwd - x)/(2*(1+num_frames_fwd))
    kernel = np.zeros((num_frames_bwd+1+num_frames_fwd), dtype=np.float)
    for i in range(num_frames_bwd):
        kernel[num_frames_bwd-i-1] = func_bwd(i)
    kernel[num_frames_bwd] = 1  # actual frame
    for i in range(num_frames_fwd):
        kernel[num_frames_bwd+1+i] = func_fwd(i)
    kernel = np.array(kernel)
    kernel = kernel/np.sum(kernel)
        
    writer = None
    h,w,c = 0,0,0

    print('Total of frames: %i, First frame: %i, Last frame: %i' % (totalFrames, firstFrame, lastFrame))
    print('Processing...')
    numFrames = int(lastFrame-firstFrame+1)
    progress_every = int(np.ceil(numFrames/10))
    progress_every = min(progress_every, 20)
    numFrames_out = 0
    for i,frame_number in enumerate(range(int(totalFrames))):
        if frame_number >= firstFrame and frame_number <= lastFrame:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
            ret, image = cap.read()
            if align: image = aligner._align_image(image)
            if i == 0:  # we need to initialize the variables
                h, w, c = image.shape
                if output_size is None:
                    output_size = (w, h)
                else:
                    if output_size_scale:
                        output_size = (round(w*output_size), round(h*output_size))
                frames_kernel = np.zeros((len(kernel), h, w, c), dtype=np.uint8)
                if output_format == 'gif':
                    print(1)
                    writer = imageio.get_writer(fname_video_out, mode='I', fps=output_fps, loop=output_loop, subrectangles=output_optimize)
                else:
                    writer = cv2.VideoWriter(fname_video_out,cv2.VideoWriter_fourcc(*output_format), output_fps, output_size, True)
            if i<(num_frames_bwd+1+num_frames_fwd):
                frames_kernel[i] = image
                continue
                
            frames_kernel = np.roll(frames_kernel, -1, axis=0)
            frames_kernel[-1] = image
            image_conv = np.zeros(image.shape, dtype=np.float)
            for ik, fac in enumerate(kernel):
                image_conv += frames_kernel[ik]*fac
            if not output_size is None:
                image_conv = cv2.resize(image_conv, output_size, interpolation=cv2.INTER_CUBIC)
            if output_format == 'gif':
                writer.append_data(cv2.cvtColor(np.round(image_conv).astype(np.uint8), cv2.COLOR_BGR2RGB))
            else:
                writer.write(np.round(image_conv).astype(np.uint8))
            numFrames_out += 1

            if update_progress:
                process_info['out_progress'] = (frame_number-firstFrame)/(lastFrame-firstFrame)
                if process_info['stop']:
                    print('\nStop signal received.')
                    break
            
            if (lastFrame-firstFrame)>0:
                if i % progress_every == 0:  # update progress every nth frame
                    _print_progressbar(frame_number-firstFrame, lastFrame-firstFrame)
                
    _print_progressbar(1,1)
    if output_format != 'gif':
        writer.release()
    print('\nNumber of frames processed: %s' % i+1)
    print('\nNumber of frames written: %s\n' % numFrames_out)
    process_info['out_ended'] = True
   

def _print_progressbar(curr, all, step=5):
    """Simple progress bar to stdout."""
    progress = round((curr)/(all)*100)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(progress/5), progress))
    sys.stdout.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Creates long exposure images from (handheld) movie recordings by aligning and stacking each frame.')
    parser.add_argument('filename', nargs='?', help='Filename of video to be loaded', default="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    import GUI
    GUI.AffexApp(parser_args=args).run()
    