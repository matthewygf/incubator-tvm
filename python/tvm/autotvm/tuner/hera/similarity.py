from scipy.spatial.distance import jaccard, cosine, euclidean, cityblock, dice

def similar_to(df, op_id, platform, features_name, target_embedding, k=5, dis=cosine, strict=True):
    temp_df = df.copy()
    if strict:
        # only compare within same operator/platform type
        temp_df = temp_df[temp_df["cand_id"] == op_id]
        temp_df = temp_df[temp_df["platform"] == platform]
    features = temp_df[features_name]
    def score(row):
        return dis(row, target_embedding)
    features["sim_score"] = features.apply(score, axis=1)
    features = features.sort_values(by="sim_score")
    
    # remove exact similar
    indexes = list(features[features["sim_score"] == 0.0].index)
    if len(indexes) > 2:
        features = features.drop(indexes[:len(indexes)-1])
    
    max_num = min(k, len(features))
    
    return df.loc[features.iloc[:max_num].index]