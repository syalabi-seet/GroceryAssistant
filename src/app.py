import streamlit as st
from embedder_core import *
from buyer_core import *

if __name__ == "__main__":  
    # Initialize states
    if 'button_clicked' not in st.session_state:
        st.session_state['button_clicked'] = False
    if 'calculate_clicked' not in st.session_state:
        st.session_state['calculate_clicked'] = False
    if 'options_state' not in st.session_state:
        st.session_state['options_state'] = None
    if 'recipe_selected' not in st.session_state:
        st.session_state['recipe_selected'] = None
    if 'cart' not in st.session_state:
        st.session_state['cart'] = None
    if 'buyer' not in st.session_state:
        st.session_state['buyer'] = None

    # Load embedder class
    embedder = load_embedder()

    st.title("Grocery Assistant")
    st.caption("Powered by Recipe1M+, Word2Vec and RPA-Python")

    # Statistics
    columns = st.columns(2)
    columns[0].metric(label="Number of unique recipes", value=1028572)
    columns[1].metric(label="Number of unique ingredients", value=13707)

    # User inputs
    columns = st.columns(2)
    with columns[0]:
        postal_code = st.text_input(
            'Please enter your postal code:', 
            placeholder='e.g. 529510')

    with columns[1]:
        top_n = st.number_input(
            label="Please adjust to see more recommendations", 
            min_value=1, max_value=10, value=5)

    # Get ingredients
    selected = st.multiselect(
        "Enter your ingredients", 
        options=embedder.vocab,
        default=[
            "cucumber",
            "fresh basil",
            "olive oil",
            "feta cheese",
            "red bell pepper"],
        help="Type and select your ingredient")

    def persist_button():
        st.session_state['button_clicked'] = True
    
    # Get document similarities
    button_state = st.button(
        'Generate recipes', 
        on_click=persist_button,
        key='Generate recipes',
        help="Press after completion of ingredient selection")
    
    if button_state or st.session_state['button_clicked']:
        with st.spinner('Processing...'):
            results = embedder.get_results(
                selected, embedder.vectors)[:top_n]

        def options_state():
            st.session_state['options_state'] = options
            st.session_state['recipe_selected'] = [x for x in results if x['title'] == options][0]

        options = st.selectbox(
            'Please select one of the recipes',
            [r['title'] for r in results],
            on_change=options_state,
            key='dropdown')
        
        if st.session_state['options_state']:
            with st.container():
                with st.form('checkbox_form'):
                    st.caption(f"URL: {st.session_state['recipe_selected']['url']}")
                    ingredients = st.session_state['recipe_selected']['raw_ingredients']
                    clean_ingredients = st.session_state['recipe_selected']['clean_ingredients']
                    final_ingredients = []
                    for i, ingredient in enumerate(ingredients):
                        if st.checkbox(ingredient, key=f"ingredient_{i}", value=True):
                            final_ingredients.append(clean_ingredients[i])
                    submit_button = st.form_submit_button('Calculate total')
                    if submit_button:
                        with st.spinner("Calculating..."):
                            if postal_code:
                                st.session_state['buyer'] = BuyerAssistant(postal_code, embedder)
                            else:
                                st.error("Please input your postal code")
                            cart, costing = st.session_state['buyer'].fetch(final_ingredients)
                            st.session_state['cart'] = cart
                            logger.debug("Calculation complete")
                            st.dataframe(st.session_state['cart'])
                            output_string = ", ".join([
                                f"Sub-total: ${costing['sub_total']:.2f}",
                                f"Delivery fee: ${costing['delivery_fee']:.2f}",
                                f"Service fee: ${costing['service_fee']:.2f}",
                                f"Total: ${costing['total']:.2f}"])
                            st.text(output_string)           

                if st.session_state['cart']:
                    if st.button('Checkout'):
                        st.session_state['buyer'].checkout()
                        st.success("Your order is complete. Thank you for shopping with us!")