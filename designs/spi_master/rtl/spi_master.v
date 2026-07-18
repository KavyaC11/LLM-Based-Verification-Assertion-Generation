// spi_master.v
// Simple SPI-lite master controller
// Supports 8-bit transfers, single clock domain, active-low reset

module spi_master (
    input  wire        clk,
    input  wire        rst_n,

    // CPU interface
    input  wire        bus_wen,       // write enable (1 = write this cycle)
    input  wire        bus_ren,       // read enable  (1 = read this cycle)
    input  wire [1:0]  bus_adr,       // register address
    input  wire [7:0]  bus_wdt,       // write data
    output reg  [7:0]  bus_rdt,       // read data
    output reg         bus_irq,       // interrupt request (active high)

    // SPI interface
    output reg         spi_sclk,      // SPI clock
    output reg         spi_mosi,      // master out slave in
    input  wire        spi_miso,      // master in slave out
    output reg         spi_cs_n       // chip select (active low)
);

// ── Register addresses ────────────────────────────────────────────────────
// 0x0 : control/status register (csr)
// 0x1 : transmit data register  (tx_data)
// 0x2 : receive data register   (rx_data)  — read only
// 0x3 : clock divider register  (clk_div)

// ── CSR bit positions ─────────────────────────────────────────────────────
// [0] go     — write 1 to start a transfer; reads 1 while transfer in progress
// [1] ien    — interrupt enable
// [2] cpol   — clock polarity (0 = idle low, 1 = idle high)
// [3] cpha   — clock phase
// [7] irq    — interrupt flag (set on transfer complete, cleared on CSR read)

// ── Internal registers ────────────────────────────────────────────────────
reg  [7:0] csr;
reg  [7:0] tx_data;
reg  [7:0] rx_data;
reg  [7:0] clk_div;

// ── FSM ───────────────────────────────────────────────────────────────────
localparam IDLE     = 2'b00;
localparam ACTIVE   = 2'b01;
localparam DONE     = 2'b10;

reg [1:0] state;
reg [2:0] bit_cnt;    // counts 0..7
reg [7:0] shift_reg;  // shift register for TX/RX
reg [7:0] clk_cnt;    // clock divider counter

// ── CSR field aliases ─────────────────────────────────────────────────────
wire go   = csr[0];
wire ien  = csr[1];
wire cpol = csr[2];
wire cpha = csr[3];

// ─────────────────────────────────────────────────────────────────────────
// CPU register write
// ─────────────────────────────────────────────────────────────────────────
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        csr     <= 8'h00;
        tx_data <= 8'h00;
        clk_div <= 8'h04;    // default divider = 4
    end else if (bus_wen) begin
        case (bus_adr)
            2'h0: csr     <= bus_wdt;
            2'h1: tx_data <= bus_wdt;
            2'h3: clk_div <= bus_wdt;
            default: ;
        endcase
    end
end

// ─────────────────────────────────────────────────────────────────────────
// CPU register read
// ─────────────────────────────────────────────────────────────────────────
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bus_rdt <= 8'h00;
    end else if (bus_ren) begin
        case (bus_adr)
            2'h0: begin
                bus_rdt <= csr;
                // reading CSR clears the irq flag
                csr[7]  <= 1'b0;
            end
            2'h2: bus_rdt <= rx_data;
            2'h3: bus_rdt <= clk_div;
            default: bus_rdt <= 8'hFF;
        endcase
    end
end

// ─────────────────────────────────────────────────────────────────────────
// Interrupt output
// ─────────────────────────────────────────────────────────────────────────
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        bus_irq <= 1'b0;
    else
        bus_irq <= csr[7] & ien;
end

// ─────────────────────────────────────────────────────────────────────────
// SPI FSM + clock generator
// ─────────────────────────────────────────────────────────────────────────
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state     <= IDLE;
        bit_cnt   <= 3'd0;
        clk_cnt   <= 8'd0;
        shift_reg <= 8'd0;
        spi_sclk  <= 1'b0;
        spi_mosi  <= 1'b0;
        spi_cs_n  <= 1'b1;
        rx_data   <= 8'h00;
        csr[0]    <= 1'b0;   // clear go
        csr[7]    <= 1'b0;   // clear irq
    end else begin
        case (state)

            IDLE: begin
                spi_cs_n <= 1'b1;
                spi_sclk <= cpol;
                if (go) begin
                    shift_reg <= tx_data;
                    bit_cnt   <= 3'd7;
                    clk_cnt   <= 8'd0;
                    spi_cs_n  <= 1'b0;
                    state     <= ACTIVE;
                end
            end

            ACTIVE: begin
                if (clk_cnt == clk_div - 1) begin
                    clk_cnt  <= 8'd0;
                    spi_sclk <= ~spi_sclk;

                    // Sample MISO on rising edge (CPOL=0, CPHA=0)
                    if (spi_sclk == cpol) begin
                        shift_reg <= {shift_reg[6:0], spi_miso};
                    end

                    // Drive MOSI on falling edge
                    if (spi_sclk != cpol) begin
                        spi_mosi <= shift_reg[7];
                        if (bit_cnt == 3'd0) begin
                            state <= DONE;
                        end else begin
                            bit_cnt <= bit_cnt - 1;
                        end
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end

            DONE: begin
                spi_cs_n  <= 1'b1;
                spi_sclk  <= cpol;
                rx_data   <= shift_reg;
                csr[0]    <= 1'b0;   // clear go — transfer complete
                csr[7]    <= 1'b1;   // set irq flag
                state     <= IDLE;
            end

            default: state <= IDLE;

        endcase
    end
end

endmodule