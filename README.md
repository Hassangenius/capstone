Crack Detection
A web app that finds cracks in photos and videos, records where each one is, and ranks governorates by how many cracks were found.

Objectives
Detect cracks automatically in photos and videos instead of relying on manual inspection.
Attach a location to every detection, down to the individual crack when the video has a GPS track.
Keep a permanent log and rank governorates by the number of cracks found.
Work from a phone, so an inspector can capture and check a crack on site.

Why use the app

Walking or driving a road to look for cracks is slow, depends on who is looking, and leaves little evidence behind. This app gives every crack the same consistent check and saves it with a time, a file name and a place.

That turns scattered observations into a record that can be sorted, mapped and compared, so repair teams can start where the damage is greatest.

The model
The app runs an Ultralytics YOLO object detector. YOLO looks at the whole image once and draws a box around each crack it finds. Several pretrained YOLO models were tested and the one that performed best on cracks was kept (best_pretrained.pt).

Input size	640 pixels
Confidence threshold	0.40. Boxes the model is less than 40% sure about are dropped.
Overlap handling	Overlapping boxes are cleaned up, and a box mostly covered by a more confident box is removed.
Video	Every third frame is analysed, and a tracker gives each crack an ID so it is counted once.
How the app works
Capture
The user uploads a photo (JPG, PNG or WebP), takes one with the phone camera, or uploads a video.

Locate
Photos use the GPS stored in the picture, then the device location. Videos use a GPX track recorded while filming, then the GPS stored in the video file, then the device location.

Detect
The YOLO model scans the image or each sampled video frame and draws boxes on the cracks it finds.

Track
In a video, each crack keeps one ID across frames, so it is logged once with its video time and its own location when a GPX track is available.

Log
Every result is saved to a local SQLite database with the time, file name, coordinates, location source and crack count.

Report
Coordinates are converted to governorate names through OpenStreetMap. The app shows a ranked table, a map with English labels, and an annotated video to download.

Using the app
Open the Image or Video tab and add a file, or take a photo.
Allow location access in the browser when asked, or add a GPX file for a video.
Press the run button and read the annotated result and the crack count.
Scroll down to see governorates ranked by cracks, the map, and the full log.
Technologies
Language
Python
Detection
Ultralytics YOLO, PyTorch
Image and video
OpenCV, Pillow
App interface
Streamlit, streamlit-js-eval for the device location
Data
pandas, NumPy, SQLite
Location and maps
EXIF and GPX data, OpenStreetMap Nominatim for governorate names, Leaflet with Esri tiles for the map
Business impact
Faster surveys
A phone and one drive can cover work that otherwise needs a team on foot, and results appear straight away.

Better priorities
Ranking governorates by cracks shows where to send crews first and where to spend the maintenance budget.

Lower repair cost
Small cracks are cheaper to seal than the damage they cause once water gets in, so finding them early saves money.

Who benefits
Municipalities and public works authorities, who get a dated and mapped record for planning and audits.
Maintenance contractors, who get a clear list of locations to visit.
Road users, who benefit from safer and better kept roads.
