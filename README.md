1. Remove broken venv  
rm -rf data/.venv

2. Create FRESH venv in project root
python3 -m venv .venv

3. Activate
source .venv/bin/activate

4. Install dependencies
python3 -m pip install -r requirements.txt

5. Run training
python3 train3.py --run "perfect-run-1"

6. In NEW terminal, view results
cd /Users/j.sudharshansharma/Desktop/vineethworkplace/mlops-wine-prediction-model/Wine-Prediction-Model
7. mlflow ui

8. Open http://localhost:5000


<img width="1416" height="375" alt="Screenshot 2026-01-03 at 7 39 38 PM" src="https://github.com/user-attachments/assets/80ceda27-254c-42d2-b93a-02a5487d1487" />


<img width="1463" height="718" alt="Screenshot 2026-01-03 at 7 17 31 PM" src="https://github.com/user-attachments/assets/370ba54f-1a69-43e4-8cb2-41bae17987e8" />
