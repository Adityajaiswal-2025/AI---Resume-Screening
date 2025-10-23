def preprocess_data(csv_path, output_path=None):
    import pandas as pd
    import re
    import os
    
    df = pd.read_csv(csv_path)
    
    df['cleaned_resume'] = df['Resume'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
    df['cleaned_resume'] = df['cleaned_resume'].apply(lambda x: x.lower().strip())

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Preprocessed data saved to {output_path}")

    return df
