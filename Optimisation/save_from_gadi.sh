finalSpeed=20
maxTime=1440
runID=Foptoelastic_regimemonochrome_LO_MMA_fixgaussian20_50GW
# runID=Fkpr_unstablemonochrome_fixgaussian20_50GW
# runID=Fasymp${finalSpeed}_fixgaussian20_50GW
# saveDir=/Users/jadonlin/Library/CloudStorage/OneDrive-TheUniversityofSydney\(Students\)/Doppler\ Damping\ -\ Jadon\ Lin/Documentation/Data/relativistic-lightsail-dynamics/Optimisation/Jadon\'s\ results/Fasymp/final_speed${finalSpeed}/maxtime${maxTime}/${runID}
# saveDir=/Users/jadonlin/Library/CloudStorage/OneDrive-TheUniversityofSydney\(Students\)/Doppler\ Damping\ -\ Jadon\ Lin/Documentation/Data/relativistic-lightsail-dynamics/Optimisation/Jadon\'s\ results/Fkpr_unstable/mono/maxtime${maxTime}/${runID}
saveDir=/Users/jadonlin/Library/CloudStorage/OneDrive-TheUniversityofSydney\(Students\)/Doppler\ Damping\ -\ Jadon\ Lin/Documentation/Data/relativistic-lightsail-dynamics/Optimisation/Jadon\'s\ results/Foptoelastic_regime/mono/maxtime${maxTime}/${runID}
mkdir -p "${saveDir}"
scp jl7180@gadi-dm.nci.org.au:~/RotationTwobox/Data/${runID}* "${saveDir}"