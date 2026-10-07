const navItems = document.querySelectorAll(".nav-item");

navItems.forEach(function (item) {

    item.addEventListener("click", function (event) {

        event.preventDefault();

        navItems.forEach(function (nav) {
            nav.classList.remove("active");
        });

        this.classList.add("active");

    });

});