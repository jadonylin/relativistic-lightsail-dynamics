The data in this folder is outdated but kept for the record.

All saved data has been moved to a OneDrive folder where it is untracked by git, but tracked by OneDrive's versioning.
This is preferable because I have found no way for git to handle large data files across local and remote repos. For
instance, git LFS almost broke the repo because it doesn't work with public forks (and it has a file size limit of 2 GB
on the free plan, which may be exceeded by some of our data files). Therefore, even if we stored our data files within the
git repo, they would be untracked. These untracked files create serious issues when pulling and pushing from multiple 
local repos, because they could result in data files overriding or mixing with each other. 

The only solution I have is to move the data to a OneDrive folder within the Doppler Damping project, which can be 
accessed using absolute file paths. All data is thus stored and updated within this OneDrive folder, meaning we have 
a common data directory for all local repos to access. 