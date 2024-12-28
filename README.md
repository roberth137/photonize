The goal of this package is anlyzing photons data from the LINCAM.

In this analysis all data is analysed prior with the Picasso package. 

To further optimize this data, a picasso file with picked localizations, the photons file with LINCAM coordiantes divided by 16 (16x binning) and the drift file from picasso are required.

The main functions are event.event_analysis() (for analysing whole binding events) and locs.locs_lt() (for analysing localizaitons).

first the corresponding modules have to be imported. Then the function can be called.

Example: 

$ import event
$ event.event_analysis('t/orig58_pf.hdf5', 't/orig58_index.hdf5', 't/orig58_drift.txt', 10, 4.5, 200, 'test')


For event analysis the imput is:

- picked localizations 
- photons 
- drift 
- offset 
- diameter of fit region (~box side length but smaller)
- integration time 
- suffix for labeling the output file 


If you have questions please let me know: rahollmann137@gmail.com

This is part of my masters project in Simoncelli lab at the London Centre for Nanotechnology.
Hope you Enjoy ;) 