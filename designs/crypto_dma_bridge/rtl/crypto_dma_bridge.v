module crypto_dma_bridge (
    input  wire        clk,
    input  wire        rst_n,
    
    // Configuration Interface
    input  wire        cfg_wen,
    input  wire [7:0]  cfg_addr,
    input  wire [31:0] cfg_wdata,
    output reg  [31:0] cfg_rdata,
    
    // DMA Master Interface
    output reg         dma_req,
    input  wire        dma_ack,
    output wire [31:0] dma_addr,
    output wire [15:0] dma_len,
    
    // Stream Output Interface
    output reg         stream_valid,
    output wire [31:0] stream_data,
    input  wire        stream_ready,
    
    // Status
    output reg         irq,
    output reg         err_flag
);

    // Internal Registers
    reg [31:0] ctrl_reg;
    reg [31:0] stat_reg;
    reg [31:0] addr_reg;
    reg [15:0] len_reg;
    reg [31:0] cipher_data;

    // FSM States
    localparam IDLE    = 3'b000;
    localparam REQUEST = 3'b001;
    localparam ACTIVE  = 3'b010;
    localparam DONE    = 3'b011;

    reg [2:0] state;
    reg [2:0] next_state;

    // Continuous Assignments
    assign dma_addr    = addr_reg;
    assign dma_len     = len_reg;
    assign stream_data = cipher_data;

    // Configuration Read Logic
    always @(*) begin
        case (cfg_addr)
            8'h00: cfg_rdata = ctrl_reg;
            8'h04: cfg_rdata = stat_reg;
            8'h08: cfg_rdata = addr_reg;
            8'h0C: cfg_rdata = {16'h0000, len_reg};
            default: cfg_rdata = 32'hFFFFFFFF;
        endcase
    end

    // Configuration Write & Interrupt/Error Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_reg <= 32'h0;
            addr_reg <= 32'h0;
            len_reg  <= 16'h0;
            err_flag <= 1'b0;
            irq      <= 1'b0;
        end else begin
            // Default clear conditions
            if (ctrl_reg[1]) irq <= 1'b0; // Clear interrupt
            
            // Error handling for rogue ACKs
            if (dma_ack && !dma_req) begin
                err_flag <= 1'b1;
                irq      <= 1'b1;
            end

            // Config Write
            if (cfg_wen) begin
                case (cfg_addr)
                    8'h00: ctrl_reg <= cfg_wdata;
                    8'h08: addr_reg <= cfg_wdata;
                    8'h0C: len_reg  <= cfg_wdata[15:0];
                    default: begin
                        // Invalid address write sets error
                        if (cfg_addr > 8'h0C) begin
                            err_flag <= 1'b1;
                            irq      <= 1'b1;
                        end
                    end
                endcase
            end

            // FSM driven interrupts
            if (state == DONE) begin
                irq <= 1'b1;
            end
        end
    end

    // FSM State Register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    // FSM Next State & Output Logic
    always @(*) begin
        next_state   = state;
        dma_req      = 1'b0;
        stream_valid = 1'b0;
        stat_reg     = 32'h0;

        case (state)
            IDLE: begin
                stat_reg[0] = 1'b0;
                if (ctrl_reg[0]) begin // Start bit
                    next_state = REQUEST;
                end
            end

            REQUEST: begin
                stat_reg[0] = 1'b1; // Busy
                dma_req     = 1'b1;
                if (dma_ack) begin
                    next_state = ACTIVE;
                end
            end

            ACTIVE: begin
                stat_reg[0]  = 1'b1; // Busy
                stream_valid = 1'b1;
                if (stream_ready) begin
                    // In reality, count down len_reg. Stubbed to just finish.
                    next_state = DONE; 
                end
            end

            DONE: begin
                stat_reg[1] = 1'b1; // Done
                next_state  = IDLE;
                // Note: irq is set in the sequential block above
            end
            
            default: next_state = IDLE;
        endcase
    end

    // Cipher Data Mock (Stable when valid but not ready)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cipher_data <= 32'h0;
        end else if (state == ACTIVE) begin
            if (stream_ready && stream_valid) begin
                cipher_data <= cipher_data + 32'h1; // Mock encryption
            end
        end
    end

endmodule