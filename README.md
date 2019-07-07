# Photo Forensics
This repository is for our Computer Vision project.

Team Members:
<ul>
<li>Shashwat Srivastava</li>
<li>Vince Pascale</li>
<li>William Bravo</li>
</ul>
In this project, we tried to implement an image security system, which:
<ol>
<li>Uses Error Level Analysis (ELA) to detect if an <b>JPEG image</b> has been digitally modified</li>
<li>Applies the Image Duplication Detection (IDD) procedure to see if a part of an image
    has been digitally copied and pasted onto the same image</li>
<li>Uses the face hallucination algorithm to make a <b> super-resolution </b> image of
    the given pixelated images [we were not able to put the code here since the sheer size of
    all the files along with the code was way too large].</li>
</ol>
<h2> Potential Application </h2>  
The way we see this project is as a image security system for law enforcement.
Suppose, a person goes missing and a third person goes to the law enforcement with an image of
taken from a CCTV footage. Now, the law enforcement needs to find the person, but taking into 
account that one will have to significantly zoom into the image to get the face, the 
face in the image of the CCTV footage is too pixelated or blurry. This is where the 
super resolution algorithm comes handy. It gives the police a sharper image of the 
missing person's face. 
But wait a second........ There's a catch here. What if someone is trying to play 
double bluff with the police and the CCTY footage image has been somehow modified?
So, before face hallucination is deployed, the ELA and IDD procedures will come in handy.


<h3> Disclaimer </h3>
This project was meant to be educational use only and uses information from various papers
and research, especially that of Hany Farid. We do not intend to use this for commercial purposes 
in any way whatsoever.
