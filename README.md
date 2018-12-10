# Photo Forensics
This repository is for our Computer Vision class final project.

Team Members:
<ol>
<li>Shashwat Srivastava</li>
<li>Vince Pascale</li>
<li>William Bravo</li>
</ol>
In this project, we tried to implement an image security system, which:
1. Uses Error Level Analysis (ELA) to detect if an <b>JPEG image</b> has been digitally modified
2. Applies the Image Duplication Detection (IDD) procedure to see if a part of an image
    has been digitally copied and pasted onto the same image
3. Uses the face hallucination algorithm to make a <b> super-resolution </b> image of
    the given pixelated images.
  
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
This project was meant to be educational use only. We do not intend to use this for 
commercial purposes in any way whatsoever.
