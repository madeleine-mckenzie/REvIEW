# REvIEW Examples

The [REvIEW.ipynb](https://github.com/madeleine-mckenzie/REvIEW/blob/main/REvIEW.ipynb) notebook comes with example values to test whether the code is working on your machine. Here I'll discuss the outputs of these example plots to illustrate how I would use the code. 

**Note:*** scroll to the end of the document for advice on removing the bad fits. 

## Fits I would accept:

### 4808.16 Fe I line:
![](https://github.com/madeleine-mckenzie/REvIEW/blob/main/Images/4808.16_fit.png)

This is an example of an unblended iron line that has been correctly fit by REvIEW. If only all the lines looked like this...

### 4789.651 Fe I line:
![](https://github.com/madeleine-mckenzie/REvIEW/blob/main/Images/4789.651_fit.png)

Even though this line is blended, REvIEW has still correctly fit this line.

### 4794.36 Fe I line:
![](https://github.com/madeleine-mckenzie/REvIEW/blob/main/Images/4794.36_fit.png)

M22_III-14 has been correctly fit but M22_III-3 is dominated by noise. Depending on the precision you want to achieve in your study, you can choose to keep or remove this line.

## Fits I would remove:

### 4798.27 Fe I line:
![](https://github.com/madeleine-mckenzie/REvIEW/blob/main/Images/4798.27_fit.png)

The line has not been correctly identified and would need to be manually removed.

### 4802.88 Fe I line:
![](https://github.com/madeleine-mckenzie/REvIEW/blob/main/Images/4802.88_fit.png)

M22_III-3 is acceptable, but the noise in M22_III-14 has meant that the fit is not as good as it could be. 

## Removing bad fits:
Whether it's because of noise, blends or magic, sometimes REvIEW just doesn't do a great job of fitting the lines (sorry, I'm trying my best to minimise this happening). There are two main approaches I have used to removing bad lines.

1) Add an additional flag to the final column (e.g. 6) and all bad fits you change the flag to 6. Then during the processing step, filter out any stars with this flag.
2) Removing the entire line from the csv file.




