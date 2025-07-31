finalSpeed=20
maxTime=1440
runID=Fasymp20_cutoffrange_50GW
saveDir=/Users/jadonlin/Library/CloudStorage/OneDrive-TheUniversityofSydney\(Students\)/Doppler\ Damping\ -\ Jadon\ Lin/Documentation/Data/relativistic-lightsail-dynamics/Optimisation/Jadon\'s\ results/Fasymp/final_speed${finalSpeed}/maxtime${maxTime}/${runID}
mkdir "${saveDir}"
scp jl7180@gadi-dm.nci.org.au:~/RotationTwobox/Data/${runID}* "${saveDir}"