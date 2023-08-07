import pandas as pd


class Recommender:
    def __init__(self, df, user_id, item_id, target, similarity, threshold=0.5):
        """
        :param df: DataFrame
        :param user_id: str
        :param item_id: str
        :param target: str
        :param similarity: str
        :param threshold: float
        """
        self.df = df
        self.user_id = user_id
        self.item_id = item_id
        self.target = target
        self.similarity = similarity
        self.threshold = threshold

    def _get_user_item_matrix(self):
        """
        :return: DataFrame
        """
        return self.df.pivot_table(index=self.user_id, columns=self.item_id, values=self.target)

    def _get_similarity_matrix(self):
        """
        :return: DataFrame
        """
        user_item_matrix = self._get_user_item_matrix()
        if self.similarity == 'cosine':
            return user_item_matrix.corr(method='pearson')
        elif self.similarity == 'jaccard':
            return user_item_matrix.corr(method='pearson')
        else:
            raise ValueError('similarity must be either "cosine" or "jaccard"')

    def _get_recommendation_matrix(self):
        """
        :return: DataFrame
        """
        user_item_matrix = self._get_user_item_matrix()
        similarity_matrix = self._get_similarity_matrix()
        return user_item_matrix.dot(similarity_matrix)

    def _get_recommendation(self, user):
        """
        :param user: str
        :return: DataFrame
        """
        recommendation_matrix = self._get_recommendation_matrix()
        user_item_matrix = self._get_user_item_matrix()
        user_items = user_item_matrix.loc[user].dropna().index
        recommendation = recommendation_matrix.loc[user].drop(user_items).sort_values(ascending=False)
        return recommendation

    def _get_recommendation_list(self, user):
        """
        :param user: str
        :return: list
        """
        recommendation = self._get_recommendation(user)
        return recommendation[recommendation > self.threshold].index.tolist()

    def _get_recommendation_df(self):
        """
        :return: DataFrame
        """
        recommendation_list = []
        for user in self.df[self.user_id].unique():
            recommendation_list.append(self._get_recommendation_list(user))
        return pd.DataFrame({self.user_id: self.df[self.user_id].unique(),
                             'recommendation': recommendation_list})
    
    def get_recommendation(self, user):
        """
        :param user: str
        :return: list
        """
        return self._get_recommendation_list(user)
    

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    recommender = Recommender(df, 'user_id', 'item_id', 'rating', 'cosine')
    print(recommender.get_recommendation('user_1'))