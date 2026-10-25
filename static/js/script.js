document.addEventListener("DOMContentLoaded", function () {
    const navItems = document.querySelectorAll(".nav-item");

    // Sidebar navigation
    navItems.forEach(function (item) {
        item.addEventListener("click", function (event) {
            const href = this.getAttribute("href");

            // Allow normal navigation for the Dashboard link
            if (href === "/" || href === null) {
                return;
            }

            // Ignore placeholder links
            if (href === "#") {
                event.preventDefault();
                return;
            }

            // Handle links pointing to page sections
            if (href.startsWith("#")) {
                const target = document.querySelector(href);

                if (target) {
                    event.preventDefault();

                    navItems.forEach(function (nav) {
                        nav.classList.remove("active");
                    });

                    this.classList.add("active");

                    target.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });
                }
            }
        });
    });

    // File upload: display the selected filename
    const fileInput = document.getElementById("fileInput");
    const selectedFile = document.getElementById("selectedFile");

    if (fileInput && selectedFile) {
        fileInput.addEventListener("change", function () {
            if (this.files.length > 0) {
                selectedFile.textContent = this.files[0].name;
            } else {
                selectedFile.textContent = "";
            }
        });
    }
});