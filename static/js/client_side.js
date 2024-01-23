$(document).ready(function(){
  
  // -[Animasi Scroll]---------------------------
  
  $(".navbar a, footer a[href='#halamanku']").on('click', function(event) {
    if (this.hash !== "") {
      event.preventDefault();
      var hash = this.hash;
      $('html, body').animate({
        scrollTop: $(hash).offset().top
      }, 900, function(){
        window.location.hash = hash;
      });
    } 
  });
  
  $(window).scroll(function() {
    $(".slideanim").each(function(){
      var pos = $(this).offset().top;
      var winTop = $(window).scrollTop();
        if (pos < winTop + 600) {
          $(this).addClass("slide");
        }
    });
  });

  
  // -[Prediksi Model]---------------------------
  
  // Fungsi untuk memanggil API ketika tombol prediksi ditekan
  $("#prediksi_submit").click(function(e) {
    e.preventDefault();
	
	// Set data pengukuran bunga iris dari input pengguna
  var input_age = $("#range_age").val(); 
	var input_sex  = $("#range_sex").val(); 
	var input_cp = $("#range_cp").val(); 
	var input_trestbps  = $("#range_trestbps").val();
  var input_chol  = $("#range_chol").val();
  var input_fbs  = $("#range_fbs").val();
  var input_restecg  = $("#range_restecg").val();
  var input_thalach  = $("#range_thalach").val();
  var input_exang  = $("#range_exang").val(); 
  var input_oldpeak  = $("#range_oldpeak").val();
  var input_slope  = $("#range_slope").val();
  var input_ca  = $("#range_ca").val();
  var input_thal  = $("#range_thal").val();

	// Panggil API dengan timeout 1 detik (1000 ms)
    setTimeout(function() {
	  try {
			$.ajax({
			  url  : "/api/deteksi",
			  type : "POST",
			  data : {"age" : input_age,
						"sex"  : input_sex,
						"cp" : input_cp,
						"trestbps"  : input_trestbps,
						"chol"  : input_chol,
						"fbs"  : input_fbs,
						"restecg"  : input_restecg,
						"thalach"  : input_thalach,
						"exang"  : input_exang,
						"oldpeak"  : input_oldpeak,
						"slope"  : input_slope,
						"ca"  : input_ca,
						"thal"  : input_thal,
			         },
			  success:function(res){
				// Ambil hasil prediksi spesies dan path gambar spesies dari API
				res_data_prediksi   = res['tulisan_prediksi']
				
				// Tampilkan hasil prediksi ke halaman web
			    generate_prediksi(res_data_prediksi); 
			  }
			});
		}
		catch(e) {
			// Jika gagal memanggil API, tampilkan error di console
			console.log("Gagal !");
			console.log(e);
		} 
    }, 1000)
    
  })
    
  // Fungsi untuk menampilkan hasil prediksi model
  function generate_prediksi(data_prediksi) {
    var str="";
    str += "<br>";
    str += "<h3>" + data_prediksi + "</h3>";
    $("#hasil_prediksi").html(str);
  }  
  
})
  
