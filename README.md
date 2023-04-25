### A streamlit web app "Process Mining training" - "Alpha Miner" module
This web app helps beginners in Process Mining understand the Process Discovery topic using the Alpha Algorithm. You can use one of the pre-installed simple event logs or create your own by modifying one. We specifically use the event log format found in Process Mining textbooks [1]: $L =[< a, c, d>^{45},< b, c, e >^{42}]$. The app includes one exercise that performs the Alpha Algorithm step by step and visualizes the discovered process model as a Petri net.     

### Usage   
You can try the app on the Streamlit Cloud via the following [URL](   ).    
Alternatively, you can download and deploy it on your computer as a Python application based on the Streamlit framework (see `requirements.txt` and `packages.txt` for details).     

### Features
This web app is designed solely for training purposes to help users understand the essential aspects of the Process Discovery using the Alpha Algorithm and learn how to create this algorithm using Python if desired. To this end:    
1. There are no options for uploading event logs from any files - only manual input or selection from the pre-installed list.    
2. Python code is easy to read - only one file, no classes, and all text information in one dictionary.    
3. All intermediate calculation results are displayed.    
4. Tables and graphs are used to visualize a Petri net.    

### Exercises
One exercise in the "Alpha Miner" app involves creating a Petri net as a process model applying the Alpha Algorithm to a simple event log. 
The app focuses on identifying the maximal candidate places as the most complex step of the Alpha Algorithm.        
     
You can do all the exercises manually and compare the results with the app.

### References
[1] van der Aalst, W.M.P.: Foundations of Process Discovery. In: van der Aalst, W.M.P., Carmona, J. (eds.) PMSS 2022. LNBIP, vol. 448, pp. 37–75. Springer, Cham (2022).
https://doi.org/10.1007/978-3-031-08848-3_2    

### Contact information
The initial version of the `pm-training-directly-follows-graph` was developed by Alexander Tolmachev.    
E-mail: axtolm@gmail.com 


