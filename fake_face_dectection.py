
import concurrent.futures
from termcolor import colored
from SpoofDetector1.web_API_spoof_attack import predict_spoof as predict_spoof1
from SpoofDetector2.web_API_spoof_attack import predict_spoof as predict_spoof2
from SpoofDetector3.web_API_spoof_attack import predict_spoof as predict_spoof3
import time


STYLE = '''
=====STYLE - TAYLOR SWIFT=====
Midnight
You come and pick me up, no headlights
Long drive
Could end in burning flames or paradise
Fade into view, oh
It's been a while since I have even heard from you (heard from you)
And I should just tell you to leave 'cause I
Know exactly where it leads, but I
Watch us go 'round and 'round each time
You got that James Dean daydream look in your eye
And I got that red lip classic thing that you like
And when we go crashing down, we come back every time
'Cause we never go out of style, we never go out of style
You got that long hair, slicked back, white T-shirt
And I got that good girl faith and a tight little skirt
And when we go crashing down, we come back every time
'Cause we never go out of style, we never go out of style
So it goes
He can't keep his wild eyes on the road, mm
Takes me home
The lights are off, he's taking off his coat, mm, yeah
I say, "I heard, oh
That you've been out and about with some other girl, some other girl"
He says, "What you heard is true, but I
Can't stop thinkin' 'bout you and I"
I said, "I've been there too a few times"
'Cause you got that James Dean daydream look in your eye
And I got that red lip classic thing that you like
And when we go crashing down, we come back every time
'Cause we never go out of style, we never go out of style
You got that long hair, slicked back, white T-shirt
And I got that good girl faith and a tight little skirt (a tight little skirt)
And when we go crashing down, we come back every time
'Cause we never go out of style (we never go), we never go out of style
Take me home
Just take me home
Yeah, just take me home
Oh, whoa, oh
(Out of style)
Oh, you got that James Dean daydream look in your eye
And I got that red lip classic thing that you like
And when we go crashing down (now we go), we come back every time
'Cause we never go out of style, we never go out of style'''
ANSI_PINK = '\033[95m'
ANSI_RESET = '\033[0m'

def predict(blinding_light):
    st = time.time()
    PREDICT_SPOOF = [predict_spoof1, predict_spoof2, predict_spoof3]
    # print(content)
    image_urls = blinding_light['image_arrays']
    
    chunk_size = 3

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    print(colored(STYLE, 'magenta'))
    all_futures=[]
    for i in range(0, len(image_urls), chunk_size):
        chunk = image_urls[i:i + chunk_size]

        futures = [executor.submit(PREDICT_SPOOF[j], url) for j, url in enumerate(chunk)]
        for future in futures:
            result = future.result()
            all_futures.append(result)
            # print(result)
    executor.shutdown(wait=True)
    print(time.time()-st)
    print({'predicted_label': all_futures})
    return {'predicted_label': all_futures}
