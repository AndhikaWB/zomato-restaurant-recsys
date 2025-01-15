import polars as pl

from typing import Callable
from collections import Counter
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator


class ColumnSelector:
    """`make_column_selector` for Polars dataframe.

    It's basically the same as `df.select(XXX).columns`.
    """

    def __init__(self, *selector: pl.Expr):
        self.selector = selector

    def __call__(self, df: pl.DataFrame):
        return df.select(*self.selector).columns


class ListJoiner(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Convert list (of string) to pure string data type.

    Args:
        sep (str, optional): Separator for the joined list element.
            Defaults to ' '.
        join_all_cols (bool, optional): Join all columns as a single
            column, useful for embedding. Defaults to False.
        copy (bool, optional): Copy input instead of modifying directly.
            Defaults to True.
    """

    def __init__(self, sep=' ', join_all_cols=False, copy=True):
        self.sep = sep
        self.join_all_cols = join_all_cols
        self.copy = copy

    def fit(self, X: pl.DataFrame, y=None):
        return self

    def transform(self, X: pl.DataFrame):
        if self.copy:
            X = X.clone()

        # Track the name of columns that will converted
        names = X.select(pl.col(pl.List(pl.String))).columns

        # Convert list columns to string columns
        X = X.with_columns(pl.col(pl.List(pl.String)).list.join(self.sep))

        if self.join_all_cols:
            # Join column names to create a new column
            names_joined = '_'.join(names)

            # Create the new joined column
            X = X.with_columns(
                pl.concat_str([pl.col(c) for c in names], separator=self.sep).alias(
                    names_joined
                )
            )

            # Delete the original columns
            X = X.drop(names)

        return X


# https://pytorch.org/text/stable/vocab.html
# https://radimrehurek.com/gensim/corpora/dictionary.html


class SentenceToIndex(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Tokenize sentence and make it as list of word index.

    Args:
        max_vocab (int, optional): Max words to keep in vocabulary
            (used together with prune). Defaults to 5000.
        batch_size (int, optional): Process only X rows per batch when
            fitting vocabulary. Defaults to 10000.
        prune_int (int, optional): Prune words in vocabulary every X
            batch interval. Defaults to 5.
        reset_on_fit (bool, optional): Reset word vocabulary before
            each `fit` call. Defaults to True.
        lowercase (bool, optional): Lowercase all words before
            fitting. Defaults to True.
        alphanum_only (bool, optional): Strips non-alphabet and
            non-number before tokenizing words. Defaults to True.
        preprocessor (Callable, optional): Custom preprocessing
            function before tokenizing words. Defaults to None.
        tokenizer (Callable, optional): Custom tokenizer function.
            Defaults to None.
        copy (bool, optional): Copy input instead of modifying
            directly. Defaults to True.
        max_words (int, optional): Max words to keep on each sentence
            when transforming. Defaults to 50.
        padding (str, optional): Word padding strategy for short
            sentence. Defaults to "start".
        unnest (bool, optional): Unnest all words in column as its
            own column when transforming. Defaults to True.
    """

    def __init__(
        self,
        max_vocab=5000,
        batch_size=10000,
        prune_int=5,
        reset_on_fit=True,
        lowercase=True,
        alphanum_only=True,
        preprocessor: Callable[[pl.Series], pl.Series] = None,
        tokenizer: Callable[[pl.Series], pl.Series] = None,
        copy=True,
        max_words: pl.DataFrame = 50,
        padding='start',
        unnest=True,
    ):
        # Fit only
        self.max_vocab = max_vocab
        self.batch_size = batch_size
        self.prune_int = prune_int
        self.reset_on_fit = reset_on_fit

        # Used in both
        self.lowercase = lowercase
        self.alphanum_only = alphanum_only
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        # Transform only
        self.copy = copy
        self.max_words = max_words
        self.padding = padding
        self.unnest = unnest

        # Vocabulary (word frequency dict)
        self.vocab = Counter()

    def __preprocess(self, col: pl.Series) -> pl.Series:
        """Clean and tokenize sentence.

        Args:
            col (pl.Series): The column to process.

        Returns:
            pl.Series: The modified column.
        """
        if self.lowercase:
            col = col.str.to_lowercase()

        if self.alphanum_only:
            col = col.str.replace_all(r'[^\w\s]+|[_]+', '')

        if self.preprocessor:
            col = self.preprocessor(col)

        if self.tokenizer:
            col = self.tokenizer(col)
        else:
            col = col.str.replace_all(r'\s+', ' ')
            col = col.str.split(' ')

        return col

    def __make_vocab(self, X: pl.DataFrame):
        """Add or update vocabulary from dataframe.

        Each column won't have a separate vocabulary, there's only one
        vocabulary for the whole dataframe.

        Args:
            X (pl.DataFrame): The dataframe to process.
        """
        # Always clone since we will be joining columns
        X = X.clone()

        names = X.select(pl.col(pl.String)).columns
        names_joined = '_'.join(names)

        # Combine all string columns into one column
        X = X.with_columns(
            pl.concat_str([pl.col(c) for c in names], separator=' ').alias(names_joined)
        )

        # Combine all string rows as a single row
        doc = X.get_column(names_joined).str.join(' ')
        # Preprocess and tokenize the string
        doc = self.__preprocess(doc)

        # Count the word frequency
        self.vocab += Counter(doc[0])

    def fit(self, X: pl.DataFrame, y=None):
        if self.reset_on_fit:
            self.vocab = Counter()

        # TODO: Check whether "iter_slices" is really needed or not
        # Just in case Polars is already smart enough to handle big data
        for i, df in enumerate(X.iter_slices(n_rows=self.batch_size)):
            self.__make_vocab(df)

            if i % self.prune_int == 0:
                # Keep only the top N words in vocab every X batches
                self.vocab = self.vocab.most_common(self.max_vocab)
                self.vocab = Counter(dict(self.vocab))

        return self

    def __zero_padding(self, X: pl.DataFrame, name: str, length: int):
        """List padding workaround for Polars.

        Args:
            X (pl.DataFrame): The dataframe to process.
            name (str): Column name to pad its values.
            length (int): Length of padding (including the words).

        Returns:
            pl.DataFrame: The modified dataframe.
        """
        if self.padding in ('start', 'left'):
            X = X.with_columns(
                pl.lit(0)
                .repeat_by(length - pl.col(name).list.len())
                .list.concat(name)
                .alias(name)
            )
        elif self.padding in ('end', 'right'):
            X = X.with_columns(
                pl.col(name).list.concat(
                    pl.lit(0).repeat_by(length - pl.col(name).list.len())
                )
            )

        return X

    def __word2idx(self, X: pl.DataFrame, vocab: list):
        """Tokenize sentence and convert words as index.

        Args:
            X (pl.DataFrame): The dataframe to process.
            vocab (list): Word list for word indexing.

        Returns:
            pl.DataFrame: The modified dataframe.
        """
        # Track the name of string columns
        names = X.select(pl.col(pl.String)).columns

        for name in names:
            col = X.get_column(name)
            # Preprocess and tokenize the string
            col = self.__preprocess(col)

            col = col.list.eval(
                # Replace word with its index in vocabulary
                pl.element().replace_strict(
                    old=vocab,
                    # Word index starts from 2 (0 = PAD, 1 = OOV)
                    new=[i + 2 for i in range(len(vocab))],
                    # Unknown words (OOV)
                    default=1,
                ),
                parallel=True,
            )

            # Truncate long list (sentence)
            col = col.list.slice(0, self.max_words)

            # Replace the original column with new value
            X = X.replace_column(X.get_column_index(name), col)

            # TODO: Check if pad is implemented in Polars
            X = self.__zero_padding(X, name, self.max_words)

            # TODO: Check if list unnest is implemented in Polars
            if self.unnest:
                # Unnest each list element as its own column
                X = X.with_columns(
                    pl.col(name)
                    .list.to_struct(fields=[str(i) for i in range(self.max_words)])
                    .struct.unnest()
                    .name.prefix(f'{name}_')
                )

                # Drop the original column after unnesting
                X = X.drop(name)

        return X

    def transform(self, X: pl.DataFrame):
        if self.copy:
            X = X.clone()

        # Keep only the top N words as list
        vocab = [i[0] for i in self.vocab.most_common(self.max_vocab)]

        # NOTE: Polars doesn't support row modification by the index
        # So we can't just use "iter_slices" like on the fit method
        # We should check if this can cause problem on big data
        X = self.__word2idx(X, vocab)

        return X


class Disable(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Disable a step in pipeline or column transformer.

    Usages:
        ```
        pipe = Pipeline([
            ('std', StandardScaler()),
            ('clust', KMeans(n_clusters = 3))
        ])

        # Disable standard scaler
        pipe.set_params('std', Disable(pipe['std']))
        # Re-enable standard scaler
        pipe.set_params('std', pipe['std'].enable())
        # Check if standard scaler is disabled
        pipe.get_params()
        ```
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def enable(self):
        return self.transformer
