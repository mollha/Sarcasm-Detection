document.addEventListener("DOMContentLoaded", function() {

    // {{ url_for('attention') }}
    // When the user ID form is submitted, POST input to /process and if the user ID exists, redirect to welcome page
	$('#form').on('submit', function(event) {
	    let main_box = $('#main_text');

		$.ajax({
			data: {
			text: main_box.val()
			},
			type: 'GET',
			url: '/attention',
			success: function (response) {
				if (response) {
					console.log(response);
				}
				else{
					console.log('no response');
				}
				return response;
			}
		});
		// HTML automatically tries to post the form, we therefore manually stop this
		event.preventDefault();
	});

});