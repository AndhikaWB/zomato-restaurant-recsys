import os
import dotenv
import polars as pl
import pydeck as pdk
import streamlit as st
import qdrant_client as qc


DATA_PATH = 'data/zomato_clustered.json'
PREPROC_PATH = 'data/model/preproc_pipeline.pkl'
MODEL_PATH = 'data/model/embedding_weight.pth'
QDRANT_COLNAME = 'zomato_restaurants'


@st.cache_data
def load_data() -> pl.DataFrame:
    return pl.read_json(DATA_PATH)


@st.cache_resource
def load_client() -> qc.QdrantClient:
    return qc.QdrantClient(url='http://localhost:6333')


@st.cache_data
def data_to_pydeck(_df: pl.DataFrame) -> pl.DataFrame:
    # Assign cluster color to dataframe for PyDeck
    df_pyd = _df.with_columns(
        pl.col(pl.List(pl.String)).list.head(3).list.join(', '),
        pl.col('cluster')
        .replace_strict(
            {
                0: [44, 162, 95, 140],
                1: [224, 243, 219, 140],
                2: [254, 178, 76, 140],
                3: [222, 45, 38, 140],
            }
        )
        .alias('color'),
    )

    # Split list to separate columns to prevent error
    df_pyd = df_pyd.to_pandas()
    df_pyd[['r', 'g', 'b', 'a']] = df_pyd['color'].to_list()
    df_pyd = df_pyd.drop('color', axis=1)

    return df_pyd


@st.cache_resource
def pydeck_map(df_pyd: pl.DataFrame) -> pdk.Deck:
    scatter_map = pdk.Layer(
        'ScatterplotLayer',
        df_pyd,
        id='zomato-map',
        get_position=['longitude', 'latitude'],
        get_fill_color=['r', 'g', 'b', 'a'],
        get_line_color=[0, 0, 0, 140],
        get_radius=10,
        radius_min_pixels=3,
        pickable=True,
        stroked=True,
    )

    init_view = pdk.ViewState(
        # Random restaurant location
        latitude=19.099001,
        longitude=72.82757,
        # Higher = zoomed-in (max: 24)
        zoom=16,
        # Up/down angle of the map
        pitch=45,
    )

    tooltip = {
        'html': '<b>Name:</b> {name}<br/>'
        + '<b>Establishment:</b> {establishment}<br/>'
        + '<b>Cuisines:</b> {cuisines}<br/>'
        + '<b>Est. Cost for 2 People:</b> {average_cost_for_two} INR<br/>'
        + '<b>Rating:</b> {aggregate_rating} ({votes} votes)<br/>'
        + '<b>Cluster - Price Range:</b> {cluster} - {price_range}<br/>'
    }

    map_style = (
        f'https://api.maptiler.com/maps/streets-v2/style.json?key={maptiler_api}'
    )

    r = pdk.Deck(
        layers=[scatter_map],
        initial_view_state=init_view,
        map_style=map_style,
        tooltip=tooltip,
    )

    return r


@st.cache_data
def data_from_qdrant(
    _client: qc.QdrantClient, collection: str, res_id: int
) -> pl.DataFrame:
    result = _client.query_points(
        collection_name=collection,
        query=res_id,
        with_payload=True,
        limit=5,
    )

    df_qdr = pl.DataFrame([i.payload for i in result.points])
    df_qdr = df_qdr.drop('url')

    return df_qdr


if __name__ == '__main__':
    dotenv.load_dotenv()
    maptiler_api = os.environ.get('MAPTILER_API_KEY')

    st.header('Zomato Restaurant Recommendation')
    st.text('Click a restaurant and wait for the recommendation to show up...')

    df = load_data()
    df_pyd = data_to_pydeck(df)

    map_pdk = pydeck_map(df_pyd)
    map_event = st.pydeck_chart(map_pdk, on_select='rerun')

    client = load_client()

    try:
        # Will error if nothing selected
        selected = map_event.selection['objects']['zomato-map'][0]

        # May take 1-3 seconds at most
        df_qdr = data_from_qdrant(
            client, collection=QDRANT_COLNAME, res_id=selected['res_id']
        )

        st.markdown(f'5 restaurants similar to **{selected["name"]}**:')
        st.dataframe(df_qdr)
    except KeyError:
        st.warning('Nothing is selected yet...')
