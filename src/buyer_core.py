import rpa as r
from rich.progress import track
from loguru import logger


class BuyerAssistant:
    def __init__(self, postal_code, embedder):
        self.postal_code = postal_code
        self.url = 'https://www.fairprice.com.sg/'
        self.embedder = embedder.model
    

    def input_postal_code(self):
        r.url(self.url)
        r.click("//div[@class='sc-176z9a9-2 jzsXqR']")
        r.type("//input[@class='sc-1x6nxah-2 hrYVGw']", self.postal_code)
        r.click("//div[@class='suggestion-item']")
        r.click("//div[@class='sc-1h9j3i1-23 ryhaZ']//button[@class='sc-1h9j3i1-11 jiKreI']")


    def go_shopping_cart(self):
        shopping_cart_button = "//a[@class='sc-isr5sq-4 UHQoa']"
        r.click(shopping_cart_button)    
        r.wait(1)


    def clear_shopping_cart(self):
        if r.present("//span[@class='sc-isr5sq-6 eXWdks']"):
            self.go_shopping_cart()
            r.wait(1)
            r.click("//button[@class='sc-2dwxu7-7 kjkypd']")  
            r.wait(1)
            r.click("//button[@class='sc-rky8kf-5 cTjlFh']")


    def get_best_item(self, ingredient):
        r.url(f'{self.url}search?query={ingredient}')
        r.wait(2)

        item_container = "".join([
            "//div[@class='sc-33fjso-0 fkzpNN productCollection']", 
            "//div[@class='sc-33fjso-6 hUhLZO']", 
            "//div[@class='sc-1plwklf-0 iknXK product-container']"])

        # Recursively handle no search matches
        back_home_button = "//button[@class='sc-e3ayae-0 ftQTQO sc-1ochxni-0 hOSYuP']"
        if r.present(back_home_button):
            ingredient = ingredient.split(" ", 1)[-1]
            self.get_best_item(ingredient)

        # Get the first 'in-stock' item recommendation
        i, j = 1, 0
        while True:
            r.wait(2)
            cart_button = "//button[@class='sc-1axwsmm-7 euWJQc']"
            add_cart_button = f"{item_container}[{i}]{cart_button}"

            if r.present(add_cart_button):
                name_element = "//span[@class='sc-1bsd7ul-1 gGWxuk']"
                get_name = f"{item_container}[{i}]{name_element}"

                price_element = "//span[@class='sc-1bsd7ul-1 gJhHzP']"
                get_price = f"{item_container}[{i}]{price_element}"

                quantity_element = "//span[@class='sc-1bsd7ul-1 LLmwF']"
                get_quantity = f"{item_container}[{i}]{quantity_element}"         
                
                best_price = r.read(get_price).strip("$")
                best_item = r.read(get_name)
                best_quantity = r.read(get_quantity).lower()
                final_cart_button = add_cart_button
                return ({
                    'Product': best_item, 
                    'Price ($)': best_price, 
                    'Quantity': best_quantity},
                    final_cart_button)
            else:
                i += 1
                j += 1
                if j == 5:
                    new_ingredient = self.embedder.most_similar(ingredient)[0][0]
                    logger.warning(f"[{ingredient}] is out of stock, retrying with [{new_ingredient}]")
                    self.get_best_item(new_ingredient)
                    


    def add_item(self, best_item, cart_button):
        r.click(cart_button)
        logger.debug(f"[{best_item['Product']}] added to cart")

        confirm_add_button = "//button[@class='sc-z104wy-0 sc-z104wy-1 efAXUk eOhKaP']"
        if r.present(confirm_add_button):
            r.click(confirm_add_button)


    def get_costing(self):
        sub_total = float(r.read("//span[@class='sc-1bsd7ul-1 kCbZO']").strip("$"))
        delivery_fee = float(r.read("//span[@class='sc-1bsd7ul-1 ejbHKP']").strip("$"))
        total = float(r.read("//div[@class='sc-1bsd7ul-1 sc-1cawz4g-17 cqTWmU jqmmlS']").strip("$"))
        service_fee = (total - sub_total - delivery_fee)
        return {
            'sub_total': sub_total,
            'delivery_fee': delivery_fee if delivery_fee else .0,
            'service_fee': service_fee if service_fee else .0,
            'total': total}


    def get_shopping_cart(self, ingredients):
        cart = []
        for ingredient in ingredients:
            best_item, cart_button = self.get_best_item(ingredient)
            if best_item not in cart:    
                self.add_item(best_item, cart_button)
                cart.append(best_item)
        self.go_shopping_cart()
        return cart


    def fetch(self, ingredients):
        r.init(headless_mode=False)
        logger.debug("Browser opened")
        self.input_postal_code()
        logger.debug("Postal code entered")
        self.clear_shopping_cart()
        logger.debug("Shopping cart emptied")
        cart = self.get_shopping_cart(ingredients)
        logger.debug("Get items")
        costing = self.get_costing()
        logger.debug("Get total")
        return cart, costing


    def checkout(self):
        checkout_button = "//button[@data-testid='linkToCheckout']"
        r.click(checkout_button)
        logger.debug("Checkout button clicked")
        r.close()