# bridge_settlement
This is the bridge settlement project development repository.

## Code Organization
* There should be in total four types of branches in this repo:
  * master: guaranteed to be free from stylistic, syntactical, and functional errors
  * test: contains all the most updated customized functional tests, please issue a **pull request** before pushing to test
  * staging: repo to test the compatibility of all the components before moving into master
  * individual-dev: names are in the form of "name-work-version", individuals members should make changes in their only subtask repo
* Always issue a **pull request** before merging any branches!

## Folder Structure
- ROOT/
    - data/
        - SOME IMAGE FOLDERS
        - e.g. calib4
        - meas/
        - robust_reg/
            - SOME IMAGE FOLDERS 
            - e.g. calib4 (where all the hough goes)
    - git_repo

SPEC
Package         Version
--------------- -------
Django          2.0
django-environ  0.4.5
pip             19.0.3
python-dateutil 2.8.0
pytz            2019.1
setuptools      39.0.1 (41.0.1 on my local server)
six             1.12.0
smtplib/requests/email (should be by default there)

