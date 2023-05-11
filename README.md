

## Commands to run

npm run conductor
	- This is the main server, running on the left hand side of vscode

npm run card_reader
	- This is the OpenCV process, running on the right hand side

## How to reset the system
1. Click into the conductor command line (left side), hit CTRL-c. Then run `npm run conductor`, or hit up arrow then enter.
2. Click into the "Found Cards" window of the OpenCV python window (icon in the start bar). Hit the letter q.
	- Alternate method: click into the card_reader command line, hit CTRL-C, then go manually close the OpenCV windows. 
3. In the card_reader command line window, rerun `npm run card_reader`

## Watch out for...
- The camera becoming misaligned. You can adjust the Region of Interest (the region the camera is clipped down to) in the "Controls" python window. ROI top, ROI left, and ROI si...or adjust the top left ROI position, and the scale of the zoom.

## Ways to observe the system
1. Errors in the consoles in VSCode. The conductor (left hand side) throws ECONNABORTED and ECONNRESET errors occasionally. Nothing to worry about. 
2. The RedisInsight application (red icon in start bar). Shows all the domain events happening in the system. Here you can see the order in which results come back from the APIs, what those results are, etc. You can publish custom card combinations by firing the "haruspex-spread-complete" event with something like ["The Wizard", "The Rogue", "The Jake"]