import numpy as np
import pickle
from twobox import TwoBox
from parameters import D1_ND
import time

Email_result = True
t_start = time.time()

## Initialise grating
grating_pitch   = 1.5384469388251338
grating_depth   = 0.5580762361523982
box1_width      = 0.10227122552871484
box2_width      = 0.07605954942866577
box_centre_dist = 0.2669020979549422
box1_eps        = 9.614975107945112
box2_eps        = 9.382304398409568
gaussian_width  = 33.916288616522735
substrate_depth = 0.17299998450776535
substrate_eps   = 9.423032644325023

wavelength      = 1.
angle           = 0.
Nx              = 100
numG            = 25
Qabs            = np.inf

grating = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width, substrate_depth, substrate_eps,
                 wavelength, angle, Nx, numG, Qabs)

## Number of lambda' points
klambda = 1000
v_final = 5/100
lambda_final = 1/D1_ND(v_final)
lambda_array = np.linspace( wavelength, lambda_final, klambda )

## Number of delta' points
kdelta = 1000
delta_max = 15 * (np.pi / 180)
delta_min = - delta_max
delta_array  = np.linspace( delta_min, delta_max, kdelta )

## Storage arrays
Q1_array            = np.zeros( (klambda, kdelta) );        Q2_array            = np.zeros( (klambda, kdelta) )
PD_Q1_delta_array   = np.zeros( (klambda, kdelta) );        PD_Q2_delta_array   = np.zeros( (klambda, kdelta) )
PD_Q1_lambda_array  = np.zeros( (klambda, kdelta) );        PD_Q2_lambda_array  = np.zeros( (klambda, kdelta) )

# Pick a row (lambda')
for i in range(klambda):
    # Go across column (delta')
    grating.wavelength  = lambda_array[i]
    for j in range(kdelta):
        grating.angle   = delta_array[j]
        # Call function
        Q1, Q2, PD_Q1_delta, PD_Q2_delta, PD_Q1_lambda, PD_Q2_lambda = grating.return_Qs_auto()
        # Store to arrays
        Q1_array[i,j] = Q1;                         Q2_array[i,j] = Q2
        PD_Q1_delta_array[i,j] = PD_Q1_delta;       PD_Q2_delta_array[i,j] = PD_Q2_delta
        PD_Q1_lambda_array[i,j] = PD_Q1_lambda;     PD_Q2_lambda_array[i,j] = PD_Q2_lambda

t_end = time.time()-t_start
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
t_end_hours = round(t_end/60**2)
print(rf"Finished in {t_end_sec} seconds, or {t_end_min} minutes, or {t_end_hours} hours!")
print(rf"#lambda: {klambda}, #delta: {kdelta}")

## Save data
pkl_fname = rf"./Data/Lookup_table_lambda_{klambda}_by_delta_{kdelta}.pkl"
data = {'Q1': Q1_array, 'Q2': Q2_array, 'PD_Q1_delta': PD_Q1_delta_array, 'PD_Q2_delta': PD_Q2_delta_array, 'PD_Q1_lambda': PD_Q1_lambda_array, 'PD_Q2_lambda': PD_Q2_lambda_array, 
         'lambda array': lambda_array, 'delta array': delta_array}
with open(pkl_fname, 'wb') as data_file:
            pickle.dump(data, data_file)

## Send email to notify end of code
if Email_result:
    # Import the following module 
    from email.mime.text import MIMEText 
    from email.mime.multipart import MIMEMultipart 
    import smtplib 
    from login import email_address, password, from_address, to

    smtp = smtplib.SMTP('smtp.gmail.com', 587) 
    smtp.ehlo() 
    smtp.starttls() 

    # Login with your email and password 
    smtp.login(email_address, password)

    def message(subject="Python Notification", 
                text="", img=None, 
                attachment=None): 
        
        # build message contents 
        msg = MIMEMultipart() 
        
        # Add Subject 
        msg['Subject'] = subject 
        
        # Add text contents 
        msg.attach(MIMEText(text)) 

        
        return msg 

    # Call the message function 
    msg = message(subject=rf"Linux: your code has finished in {t_end_sec} seconds, or {t_end_min} minutes, or {t_end_hours} hours!" ,
                  text=rf"Parameters: #lambda: {klambda}, #delta: {kdelta}",img=None, 
                  attachment=None) 


    # Provide some data to the sendmail function! 
    smtp.sendmail(from_addr= from_address, 
                to_addrs=to, msg=msg.as_string()) 

    # Finally, don't forget to close the connection 
    smtp.quit()